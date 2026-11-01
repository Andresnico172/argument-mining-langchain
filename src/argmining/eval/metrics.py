"""Scoring a predicted ArgumentMap against a gold AAE essay.

The pipeline returns unit *texts*, not offsets, so each predicted unit is first
located in the essay and then aligned one-to-one with gold components by
character overlap. Following Eger et al. (2017), a predicted span matches a gold
span when ``overlap / max(len_pred, len_gold) >= threshold``: 0.5 is the usual
"approximate match", 1.0 an exact match.

Three end-to-end scores, micro-averaged over essays:

- ``segments``: was the component found at all (type ignored)?
- ``<role>`` / ``components_macro``: found *and* given the right role.
- ``relations`` / ``links``: a relation whose endpoints both match gold components
  and that exists in the gold annotation, with (``relations``) or without
  (``links``) the right support/attack type. A gold claim's stance points to
  "the major claim", so linking it to any of the essay's major claims counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from ..schemas import ArgumentMap
from .aae import ROLE_OF, GoldEssay

ROLES = ("major_claim", "claim", "premise")


@dataclass
class Counts:
    tp: int = 0
    n_pred: int = 0
    n_gold: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.n_pred + other.n_pred, self.n_gold + other.n_gold)

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def locate(unit: str, text: str, cursor: int = 0) -> tuple[int, int] | None:
    """Character span of ``unit`` in ``text``; tolerant of small edits by the model."""
    unit = unit.strip()
    if not unit:
        return None
    for start in (text.find(unit, cursor), text.find(unit)):
        if start >= 0:
            return start, start + len(unit)
    # Not verbatim: anchor on the longest common block, if it covers most of the unit.
    m = SequenceMatcher(None, text, unit, autojunk=False).find_longest_match(0, len(text), 0, len(unit))
    if m.size < 0.6 * len(unit):
        return None
    start = max(0, m.a - m.b)
    return start, min(len(text), start + len(unit))


def overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(a[1] - a[0], b[1] - b[0])


def align(pred: ArgumentMap, gold: GoldEssay, threshold: float = 0.5) -> dict[str, str]:
    """One-to-one map from predicted unit ids to gold component ids (greedy by overlap)."""
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for seg in pred.segments:
        span = locate(seg.text, gold.text, cursor)
        if span:
            spans[seg.id] = span
            cursor = span[1]

    candidates = [
        (overlap_ratio(span, (g.start, g.end)), pid, g.id)
        for pid, span in spans.items()
        for g in gold.components
    ]
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for ratio, pid, gid in sorted(candidates, reverse=True):
        if ratio < threshold:
            break
        if pid not in mapping and gid not in used:
            mapping[pid] = gid
            used.add(gid)
    return mapping


def score_essay(pred: ArgumentMap, gold: GoldEssay, threshold: float = 0.5) -> dict[str, Counts]:
    mapping = align(pred, gold, threshold)
    gold_role = {c.id: ROLE_OF[c.type] for c in gold.components}

    scores = {"segments": Counts(len(mapping), len(pred.segments), len(gold.components))}
    for role in ROLES:
        scores[role] = Counts(
            tp=sum(1 for pid, gid in mapping.items() if pred.roles.get(pid) == role == gold_role[gid]),
            n_pred=sum(1 for r in pred.roles.values() if r == role),
            n_gold=sum(1 for r in gold_role.values() if r == role),
        )

    typed = {(r.source, r.target, r.type) for r in gold.relations}
    untyped = {(r.source, r.target) for r in gold.relations}
    hit_typed: set[tuple] = set()
    hit_untyped: set[tuple] = set()
    for rel in pred.relations:
        src, tgt = mapping.get(rel.source), mapping.get(rel.target)
        if not src or not tgt:
            continue
        for target in (tgt, "MC") if tgt in gold.major_claims else (tgt,):
            if (src, target) in untyped:
                hit_untyped.add((src, target))
            if (src, target, rel.type) in typed:
                hit_typed.add((src, target, rel.type))

    n_pred = len(pred.relations)
    scores["links"] = Counts(len(hit_untyped), n_pred, len(untyped))
    scores["relations"] = Counts(len(hit_typed), n_pred, len(typed))
    return scores


def aggregate(per_essay: list[dict[str, Counts]]) -> dict[str, Counts]:
    total: dict[str, Counts] = {}
    for scores in per_essay:
        for name, counts in scores.items():
            total[name] = total.get(name, Counts()) + counts
    return total


def macro_f1(total: dict[str, Counts]) -> float:
    return sum(total[r].f1 for r in ROLES) / len(ROLES)
