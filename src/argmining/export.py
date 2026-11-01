"""Turn an ArgumentMap into a graph, Mermaid and Graphviz diagrams and a Markdown report."""

from __future__ import annotations

import textwrap

import networkx as nx

from .schemas import ArgumentMap

ROLE_LABEL = {"major_claim": "Major claim", "claim": "Claim", "premise": "Premise"}


def to_networkx(amap: ArgumentMap) -> nx.DiGraph:
    graph = nx.DiGraph()
    for s in amap.segments:
        graph.add_node(s.id, text=s.text, role=amap.roles.get(s.id))
    for r in amap.relations:
        graph.add_edge(r.source, r.target, type=r.type)
    return graph


def _mermaid_label(text: str, limit: int = 70) -> str:
    short = text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
    return short.replace('"', "#quot;")


def to_mermaid(amap: ArgumentMap) -> str:
    lines = ["graph BT"]
    for s in amap.segments:
        role = amap.roles.get(s.id, "unlabelled")
        lines.append(f'    {s.id}["<b>{s.id}</b> · {ROLE_LABEL.get(role, role)}<br/>{_mermaid_label(s.text)}"]:::{role}')
    for r in amap.relations:
        arrow = "-->|support|" if r.type == "support" else "-.->|attack|"
        lines.append(f"    {r.source} {arrow} {r.target}")
    lines += [
        "    classDef major_claim fill:#1f4e79,color:#fff,stroke:#1f4e79",
        "    classDef claim fill:#d6e4f0,color:#111,stroke:#1f4e79",
        "    classDef premise fill:#f4f4f4,color:#111,stroke:#888",
        "    classDef unlabelled fill:#fff,color:#888,stroke:#ccc,stroke-dasharray:3",
    ]
    return "\n".join(lines)


DOT_STYLE = {
    "major_claim": 'fillcolor="#1f4e79", fontcolor="white", color="#1f4e79"',
    "claim": 'fillcolor="#d6e4f0", fontcolor="#111111", color="#1f4e79"',
    "premise": 'fillcolor="#f4f4f4", fontcolor="#111111", color="#888888"',
}


def _dot_label(text: str, width: int = 38, lines: int = 4) -> str:
    wrapped = textwrap.wrap(text, width) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1].rstrip(" .,;") + "…"
    return "\\n".join(w.replace("\\", "\\\\").replace('"', '\\"') for w in wrapped)


def to_dot(amap: ArgumentMap) -> str:
    """Graphviz DOT version of the map (Streamlit renders it natively)."""
    lines = [
        "digraph argument_map {",
        "    rankdir=BT; nodesep=0.35; ranksep=0.55;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.15,0.08"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]
    for s in amap.segments:
        role = amap.roles.get(s.id, "unlabelled")
        style = DOT_STYLE.get(role, 'fillcolor="white", fontcolor="#888888", color="#cccccc", style="rounded,dashed"')
        header = f"{s.id} · {ROLE_LABEL.get(role, role)}"
        lines.append(f'    {s.id} [label="{header}\\n\\n{_dot_label(s.text)}", {style}];')
    for r in amap.relations:
        if r.type == "support":
            lines.append(f'    {r.source} -> {r.target} [label=" support", color="#2e7d32", fontcolor="#2e7d32"];')
        else:
            lines.append(
                f'    {r.source} -> {r.target} [label=" attack", color="#c62828", fontcolor="#c62828", style=dashed];'
            )
    lines.append("}")
    return "\n".join(lines)


def to_markdown(amap: ArgumentMap) -> str:
    out = ["# Argument analysis", ""]

    out += ["## Conclusions", ""]
    for s in amap.conclusions:
        out.append(f"- **{s.id}** ({ROLE_LABEL[amap.roles[s.id]]}): {s.text}")
    if not amap.conclusions:
        out.append("_No conclusions were identified._")

    schemes = {s.argument_id: s for s in amap.schemes}
    scores = {s.argument_id: s for s in amap.evaluation.scores} if amap.evaluation else {}
    out += ["", "## Arguments", ""]
    for arg in amap.arguments:
        out.append(f"### {arg.id} → {arg.conclusion}: {amap.text_of(arg.conclusion)}")
        out += [f"- supported by **{p}**: {amap.text_of(p)}" for p in arg.premises]
        out += [f"- attacked by **{a}**: {amap.text_of(a)}" for a in arg.attackers]
        if sch := schemes.get(arg.id):
            out.append(f"- *Scheme*: `{sch.scheme}`")
            out += [f"- *Implicit premise*: {p}" for p in sch.implicit_premises]
            out += [f"- *Critical question*: {q}" for q in sch.critical_questions]
        if sc := scores.get(arg.id):
            out.append(f"- *Strength*: {sc.strength}/5. {sc.rationale}")
        out.append("")

    if amap.evaluation:
        out += ["## Fallacies", ""]
        for f in amap.evaluation.fallacies:
            out.append(f"- `{f.name}` in {', '.join(f.unit_ids)}: {f.explanation}")
        if not amap.evaluation.fallacies:
            out.append("_None detected._")
        out += ["", "## Overall assessment", "", amap.evaluation.overall or "_n/a_", ""]

    out += ["## Pipeline", "", f"- Extraction revisions: {amap.revisions}"]
    if amap.unresolved_issues:
        out.append("- Issues still open after the revision budget:")
        out += [f"  - {i}" for i in amap.unresolved_issues]

    out += ["", "## Argument map", "", "```mermaid", to_mermaid(amap), "```", ""]
    return "\n".join(out)
