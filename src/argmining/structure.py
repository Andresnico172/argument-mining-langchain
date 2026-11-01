"""Deterministic checks and transformations over an extraction.

Nothing here calls a model: these are cheap, exact checks that run before the
LLM reviewer, so obvious mistakes are sent back without spending a model call.
"""

from __future__ import annotations

import networkx as nx

from .schemas import Argument, ExtractionOutput, Segment


def validate_extraction(segments: list[Segment], extraction: ExtractionOutput) -> list[str]:
    """Return a list of structural problems; empty when the extraction is well formed."""
    ids = [s.id for s in segments]
    known = set(ids)
    issues: list[str] = []

    seen: set[str] = set()
    graph = nx.DiGraph()
    for u in extraction.units:
        if u.id not in known:
            issues.append(f"Annotation for unknown unit '{u.id}'.")
            continue
        if u.id in seen:
            issues.append(f"Unit {u.id} was annotated more than once.")
            continue
        seen.add(u.id)

        if u.role == "major_claim":
            if u.target:
                issues.append(f"{u.id} is the major claim, so its target and relation must be null.")
            continue
        if not u.target or not u.relation:
            issues.append(f"{u.id} is a {u.role}: give the id of the unit it supports or attacks, and the relation.")
        elif u.target not in known:
            issues.append(f"{u.id} points to unknown unit '{u.target}'.")
        elif u.target == u.id:
            issues.append(f"{u.id} cannot support or attack itself.")
        else:
            graph.add_edge(u.id, u.target)

    missing = [i for i in ids if i not in seen]
    if missing:
        issues.append(f"Units without an annotation: {', '.join(missing)}.")

    roles = extraction.role_map
    if not any(roles.get(i) in ("major_claim", "claim") for i in seen):
        issues.append("No conclusion was identified: at least one unit must be a major_claim or claim.")

    try:
        cycle = nx.find_cycle(graph)
        path = " -> ".join([u for u, _ in cycle] + [cycle[0][0]])
        issues.append(f"Circular structure: {path}.")
    except nx.NetworkXNoCycle:
        pass

    return issues


def build_arguments(segments: list[Segment], extraction: ExtractionOutput) -> list[Argument]:
    """Group relations into arguments: one per unit that receives support or attack."""
    known = {s.id for s in segments}
    incoming: dict[str, tuple[list[str], list[str]]] = {}
    for rel in extraction.relations:
        if rel.source not in known or rel.target not in known or rel.source == rel.target:
            continue
        premises, attackers = incoming.setdefault(rel.target, ([], []))
        bucket = premises if rel.type == "support" else attackers
        if rel.source not in bucket:
            bucket.append(rel.source)

    order = [s.id for s in segments]
    targets = sorted(incoming, key=order.index)
    return [
        Argument(id=f"ARG{n}", conclusion=t, premises=incoming[t][0], attackers=incoming[t][1])
        for n, t in enumerate(targets, start=1)
    ]
