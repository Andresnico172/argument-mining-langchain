import json

import pytest

from argmining.eval.aae import load_essays, parse_ann
from argmining.eval.metrics import aggregate, locate, macro_f1, score_essay
from argmining.eval.run import main as eval_main
from argmining.schemas import ArgumentMap, Relation, Segment

TEXT = (
    "Cars in cities\n\n"
    "I believe cities should ban cars from their centres. First, traffic is the main source of "
    "urban air pollution. Some argue that shops would lose customers. However, pedestrian streets "
    "attract more visitors. In conclusion, car-free centres are the better choice."
)

GOLD = [
    ("T1", "MajorClaim", "cities should ban cars from their centres"),
    ("T2", "Claim", "traffic is the main source of urban air pollution"),
    ("T3", "Claim", "shops would lose customers"),
    ("T4", "Premise", "pedestrian streets attract more visitors"),
    ("T5", "MajorClaim", "car-free centres are the better choice"),
]


def make_ann() -> str:
    lines = []
    for tid, kind, span in GOLD:
        start = TEXT.index(span)
        lines.append(f"{tid}\t{kind} {start} {start + len(span)}\t{span}")
    lines += ["A1\tStance T2 For", "A2\tStance T3 Against", "R1\tattacks Arg1:T4 Arg2:T3\t"]
    return "\n".join(lines) + "\n"


def make_prediction() -> ArgumentMap:
    units = [
        ("A1", "I believe cities should ban cars from their centres.", "major_claim", None),
        ("A2", "First, traffic is the main source of urban air pollution.", "claim", ("support", "A1")),
        ("A3", "Some argue that shops would lose customers.", "premise", ("attack", "A1")),
        ("A4", "However, pedestrian streets attract more visitors.", "premise", ("support", "A3")),
        ("A5", "In conclusion, car-free centres are the better choice.", "claim", ("support", "A1")),
        ("A6", "A sentence the model made up entirely.", "premise", ("support", "A1")),
    ]
    return ArgumentMap(
        segments=[Segment(id=i, text=t) for i, t, _, _ in units],
        roles={i: r for i, _, r, _ in units},
        relations=[Relation(source=i, target=rel[1], type=rel[0]) for i, _, _, rel in units if rel],
        arguments=[], schemes=[], evaluation=None, revisions=0, unresolved_issues=[],
    )


@pytest.fixture
def gold():
    return parse_ann(TEXT, make_ann(), "essay001")


def test_parse_ann_turns_claim_stance_into_relations(gold):
    assert [c.type for c in gold.components] == ["MajorClaim", "Claim", "Claim", "Premise", "MajorClaim"]
    assert gold.major_claims == {"T1", "T5"}
    rels = {(r.source, r.target, r.type) for r in gold.relations}
    assert rels == {("T4", "T3", "attack"), ("T2", "MC", "support"), ("T3", "MC", "attack")}


def test_scores(gold):
    s = score_essay(make_prediction(), gold)
    as_tuple = lambda c: (c.tp, c.n_pred, c.n_gold)  # noqa: E731
    assert as_tuple(s["segments"]) == (5, 6, 5)
    assert as_tuple(s["major_claim"]) == (1, 1, 2)  # T5 restates the thesis but was labelled claim
    assert as_tuple(s["claim"]) == (1, 2, 2)
    assert as_tuple(s["premise"]) == (1, 3, 1)  # A3 is a gold claim; A6 matches nothing
    assert as_tuple(s["links"]) == (3, 5, 3)
    assert as_tuple(s["relations"]) == (2, 5, 3)  # A4 -> A3 has the wrong type


def test_exact_match_threshold_is_stricter(gold):
    assert score_essay(make_prediction(), gold, threshold=1.0)["segments"].tp == 0


def test_locate_tolerates_small_edits():
    span = locate("traffic is the main source of urban air polution", TEXT)
    assert span and abs(span[0] - TEXT.index("traffic")) <= 2
    assert locate("nothing like this appears anywhere", TEXT) is None


def test_aggregate_and_macro_f1(gold):
    s = score_essay(make_prediction(), gold)
    total = aggregate([s, s])
    assert total["segments"].tp == 10 and total["segments"].f1 == pytest.approx(s["segments"].f1)
    assert 0 < macro_f1(total) < 1


def test_runner_scores_cached_predictions(tmp_path):
    corpus = tmp_path / "aae2" / "brat-project-final"
    corpus.mkdir(parents=True)
    for eid in ("essay001", "essay002"):
        (corpus / f"{eid}.txt").write_text(TEXT, encoding="utf-8")
        (corpus / f"{eid}.ann").write_text(make_ann(), encoding="utf-8")
    (tmp_path / "aae2" / "train-test-split.csv").write_text(
        '"ID";"SET"\n"essay001";"TEST"\n"essay002";"TRAIN"\n', encoding="utf-8"
    )
    assert [e.id for e in load_essays(tmp_path / "aae2", split="TEST")] == ["essay001"]

    out = tmp_path / "eval"
    (out / "predictions").mkdir(parents=True)
    record = {"essay": "essay001", "model": "fake", "seconds": 1.5, "result": make_prediction().model_dump()}
    (out / "predictions" / "essay001.json").write_text(json.dumps(record), encoding="utf-8")

    eval_main(["--data", str(tmp_path / "aae2"), "--out", str(out), "--score-only", "--model", "fake"])
    summary = json.loads((out / "metrics.json").read_text())
    assert summary["essays_scored"] == 1
    assert summary["scores"]["links"]["recall"] == 1.0
    assert "| Relations (support/attack) |" in (out / "metrics.md").read_text()
