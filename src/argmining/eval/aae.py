"""Loader for the Argument Annotated Essays v2 corpus (brat standoff format).

Each essay ``essayNNN`` comes as a ``.txt`` file and a ``.ann`` file with lines like::

    T1	MajorClaim 503 575	we should attach more importance to cooperation ...
    T3	Claim 591 714	through cooperation, children can learn ...
    A1	Stance T3 For
    R1	supports Arg1:T4 Arg2:T3

Premises are linked to claims or premises with ``supports``/``attacks``.
Claims have no explicit relation: their ``Stance`` (For/Against) is towards the
major claim(s). An essay usually has several major claims (the thesis restated
in the introduction and the conclusion).

The corpus licence forbids redistribution, so it is not shipped with this
project: download it from https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

GoldType = Literal["MajorClaim", "Claim", "Premise"]

# Our role names for the corpus component types.
ROLE_OF: dict[str, str] = {"MajorClaim": "major_claim", "Claim": "claim", "Premise": "premise"}


@dataclass
class GoldComponent:
    id: str
    type: GoldType
    start: int
    end: int
    text: str
    stance: Literal["For", "Against"] | None = None


@dataclass
class GoldRelation:
    source: str
    target: str  # a component id, or "MC" for "any major claim"
    type: Literal["support", "attack"]


@dataclass
class GoldEssay:
    id: str
    text: str
    components: list[GoldComponent]
    relations: list[GoldRelation] = field(default_factory=list)

    @property
    def major_claims(self) -> set[str]:
        return {c.id for c in self.components if c.type == "MajorClaim"}


def parse_ann(text: str, ann: str, essay_id: str = "") -> GoldEssay:
    components: dict[str, GoldComponent] = {}
    relations: list[GoldRelation] = []
    stances: dict[str, str] = {}

    for line in ann.splitlines():
        if not line.strip():
            continue
        tag, body, *rest = line.split("\t")
        if tag.startswith("T"):
            kind, start, end = body.split()[:3]
            if kind not in ROLE_OF:
                continue
            s, e = int(start), int(end)
            components[tag] = GoldComponent(tag, kind, s, e, text[s:e])
        elif tag.startswith("A"):
            _, target, value = body.split()
            stances[target] = value
        elif tag.startswith("R"):
            kind, arg1, arg2 = body.split()
            relations.append(
                GoldRelation(
                    source=arg1.split(":")[1],
                    target=arg2.split(":")[1],
                    type="support" if kind == "supports" else "attack",
                )
            )

    for cid, stance in stances.items():
        if cid in components:
            components[cid].stance = stance  # type: ignore[assignment]
            if components[cid].type == "Claim":
                relations.append(
                    GoldRelation(source=cid, target="MC", type="support" if stance == "For" else "attack")
                )

    ordered = sorted(components.values(), key=lambda c: c.start)
    return GoldEssay(id=essay_id, text=text, components=ordered, relations=relations)


def find_corpus_dir(root: Path) -> Path:
    """Return the folder holding the essayNNN.ann files, searching below ``root``."""
    hit = next(iter(sorted(root.rglob("essay001.ann"))), None)
    if hit is None:
        raise FileNotFoundError(
            f"No essay*.ann files under {root}. Unzip ArgumentAnnotatedEssays-2.0.zip "
            "and the brat-project-final.zip inside it there."
        )
    return hit.parent


def load_split(root: Path) -> dict[str, str]:
    """Map essay id -> 'TRAIN' / 'TEST' from train-test-split.csv, if present."""
    hit = next(iter(root.rglob("train-test-split.csv")), None)
    if hit is None:
        return {}
    with hit.open(encoding="utf-8") as f:
        return {row["ID"]: row["SET"] for row in csv.DictReader(f, delimiter=";")}


def load_essays(root: Path, split: str | None = "TEST", limit: int | None = None) -> list[GoldEssay]:
    corpus = find_corpus_dir(root)
    splits = load_split(root)
    essays = []
    for ann_path in sorted(corpus.glob("essay*.ann")):
        essay_id = ann_path.stem
        if split and splits and splits.get(essay_id) != split:
            continue
        text = ann_path.with_suffix(".txt").read_text(encoding="utf-8")
        essays.append(parse_ann(text, ann_path.read_text(encoding="utf-8"), essay_id))
        if limit and len(essays) >= limit:
            break
    return essays
