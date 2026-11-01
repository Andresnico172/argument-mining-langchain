"""Streamlit front end. Start it with ``uv run argmine-app``.

Everything runs locally against Ollama: no API keys, no external services.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import streamlit as st
from langchain_ollama import ChatOllama

from argmining.export import ROLE_LABEL, to_dot, to_markdown, to_mermaid
from argmining.graph import build_graph
from argmining.schemas import ArgumentMap

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
DEFAULT_URL = "http://localhost:11434"

NODE_LABEL = {
    "segmenter": "Segmenter split the text into argumentative units",
    "extractor": "Extractor annotated roles and relations",
    "reviewer": "Reviewer checked the structure",
    "scheme_analyst": "Scheme analyst identified argumentation schemes",
    "evaluator": "Evaluator checked fallacies and argument strength",
    "synthesizer": "Synthesizer built the argument map",
}

PIPELINE_DOT = """digraph pipeline {
    rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f4f4", fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9, color="#666666"];
    text [shape=plaintext, fillcolor=none, label="text"];
    map [shape=plaintext, fillcolor=none, label="argument map"];
    segmenter [label="Segmenter\\nLLM"]; extractor [label="Extractor\\nLLM"];
    reviewer [label="Reviewer\\nrules + LLM", shape=diamond, style=filled];
    scheme_analyst [label="Scheme analyst\\nLLM"]; evaluator [label="Evaluator\\nLLM"];
    synthesizer [label="Synthesizer\\ncode"];
    text -> segmenter -> extractor -> reviewer;
    reviewer -> extractor [label="issues", style=dashed, constraint=false];
    reviewer -> scheme_analyst [label="ok"]; reviewer -> evaluator [label="ok"];
    scheme_analyst -> synthesizer; evaluator -> synthesizer; synthesizer -> map;
}"""


# --- Ollama helpers ------------------------------------------------------------


@st.cache_data(ttl=15, show_spinner=False)
def ollama_models(base_url: str) -> list[str] | None:
    """Installed chat models, or None when Ollama is not reachable."""
    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return None
    return sorted(m["name"] for m in tags.get("models", []) if "embed" not in m["name"])


@st.cache_resource(show_spinner=False)
def pipeline(model: str, base_url: str, num_ctx: int, max_revisions: int):
    llm = ChatOllama(model=model, base_url=base_url, temperature=0, num_ctx=num_ctx)
    return build_graph(llm, max_revisions=max_revisions)


# --- Input callbacks -----------------------------------------------------------


def load_example(path: Path) -> None:
    st.session_state.text = path.read_text(encoding="utf-8")
    st.session_state.pop("result", None)


def load_upload() -> None:
    upload = st.session_state.get("upload")
    if upload is not None:
        st.session_state.text = upload.getvalue().decode("utf-8", errors="replace")
        st.session_state.pop("result", None)


# --- Analysis ------------------------------------------------------------------


def run_analysis(text: str, model: str, base_url: str, num_ctx: int, max_revisions: int) -> None:
    app = pipeline(model, base_url, num_ctx, max_revisions)
    state: dict = {}
    start = time.perf_counter()
    with st.status(f"Analysing with {model}… (a few minutes on a laptop)", expanded=True) as status:
        try:
            for update in app.stream({"text": text}, stream_mode="updates"):
                for node, delta in update.items():
                    delta = delta or {}
                    st.markdown(f":green[:material/check_circle:] **{NODE_LABEL[node]}** · {time.perf_counter() - start:.0f} s")
                    if node == "segmenter":
                        st.caption(f"{len(delta.get('segments', []))} units · language: {delta.get('language', '?')}")
                    if node == "reviewer" and delta.get("feedback"):
                        issues = "\n".join(f"- {i}" for i in delta["feedback"])
                        st.caption(f"Sent back to the extractor:\n{issues}")
                    state.update(delta)
        except Exception as exc:  # surface Ollama/model errors in the UI instead of a traceback
            status.update(label="The analysis failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return
        seconds = time.perf_counter() - start
        status.update(label=f"Done in {seconds:.0f} s", state="complete", expanded=False)
    st.session_state.result = state["result"]
    st.session_state.seconds = seconds


# --- Result views --------------------------------------------------------------


def stars(n: int) -> str:
    return "●" * n + "○" * (5 - n)


def show_conclusions(r: ArgumentMap) -> None:
    if not r.conclusions:
        st.info("No conclusions were identified.")
    for s in r.conclusions:
        if r.roles[s.id] == "major_claim":
            with st.container(border=True):
                st.caption(f"{s.id} · MAJOR CLAIM")
                st.markdown(f"#### {s.text}")
        else:
            stance = next((rel.type for rel in r.relations if rel.source == s.id), None)
            badge = {"support": ":green-badge[supports]", "attack": ":red-badge[attacks]"}.get(stance, "")
            st.markdown(f"**{s.id}** · claim {badge}  \n{s.text}")


def show_arguments(r: ArgumentMap) -> None:
    schemes = {s.argument_id: s for s in r.schemes}
    scores = {s.argument_id: s for s in r.evaluation.scores} if r.evaluation else {}
    if not r.arguments:
        st.info("No arguments were found.")
    for arg in r.arguments:
        role = ROLE_LABEL.get(r.roles.get(arg.conclusion, ""), "")
        with st.expander(f"**{arg.id}** → {arg.conclusion} ({role}): {r.text_of(arg.conclusion)}", expanded=True):
            for p in arg.premises:
                st.markdown(f":green[:material/add_circle: **{p}**] {r.text_of(p)}")
            for a in arg.attackers:
                st.markdown(f":red[:material/do_not_disturb_on: **{a}**] {r.text_of(a)}")
            left, right = st.columns(2)
            if sch := schemes.get(arg.id):
                with left:
                    st.markdown(f"**Scheme** :blue-badge[{sch.scheme.replace('_', ' ')}]")
                    for p in sch.implicit_premises:
                        st.markdown(f"- *Implicit premise:* {p}")
                with right:
                    for q in sch.critical_questions:
                        st.markdown(f"- {q}")
            if sc := scores.get(arg.id):
                st.markdown(f"**Strength** `{stars(sc.strength)}` {sc.strength}/5 · {sc.rationale}")


def show_evaluation(r: ArgumentMap) -> None:
    if r.evaluation is None:
        st.info("No evaluation available.")
        return
    st.subheader("Fallacies")
    for f in r.evaluation.fallacies:
        quoted = " ".join(f"“{r.text_of(u)}”" for u in f.unit_ids)
        st.warning(f"**{f.name.replace('_', ' ')}** in {', '.join(f.unit_ids)}: {quoted}\n\n{f.explanation}")
    if not r.evaluation.fallacies:
        st.success("No fallacies detected.")
    st.subheader("Overall assessment")
    st.write(r.evaluation.overall or "n/a")
    if r.unresolved_issues:
        with st.expander(f"Reviewer issues still open after {r.revisions} revisions"):
            st.markdown("\n".join(f"- {i}" for i in r.unresolved_issues))


def show_export(r: ArgumentMap) -> None:
    cols = st.columns(4)
    files = [
        ("report.md", to_markdown(r), "text/markdown"),
        ("result.json", json.dumps(r.model_dump(), ensure_ascii=False, indent=2), "application/json"),
        ("argument_map.mmd", to_mermaid(r), "text/plain"),
        ("argument_map.dot", to_dot(r), "text/plain"),
    ]
    for col, (name, data, mime) in zip(cols, files):
        col.download_button(name, data, file_name=name, mime=mime, icon=":material/download:", on_click="ignore")
    with st.expander("Raw result"):
        st.json(r.model_dump(), expanded=False)


def show_result(r: ArgumentMap) -> None:
    fallacies = len(r.evaluation.fallacies) if r.evaluation else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Units", len(r.segments))
    c2.metric("Arguments", len(r.arguments))
    c3.metric("Fallacies", fallacies)
    c4.metric("Revisions", r.revisions)
    c5.metric("Time", f"{st.session_state.get('seconds', 0):.0f} s")

    tabs = st.tabs(["Conclusions", "Argument map", "Arguments", "Evaluation", "Export"])
    with tabs[0]:
        show_conclusions(r)
    with tabs[1]:
        st.graphviz_chart(to_dot(r), width="stretch")
        st.caption("Green arrows support, red dashed arrows attack. Arrows point from the reason to what it backs.")
    with tabs[2]:
        show_arguments(r)
    with tabs[3]:
        show_evaluation(r)
    with tabs[4]:
        show_export(r)


# --- Page ----------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Argument Miner", page_icon=":material/account_tree:", layout="wide")

    with st.sidebar:
        st.header("Model")
        base_url = st.text_input("Ollama URL", DEFAULT_URL)
        models = ollama_models(base_url)
        if models is None:
            st.error("Ollama is not reachable. Start it with `ollama serve` or open the Ollama app.")
            model = None
        elif not models:
            st.warning("Ollama is running but has no models. Run `ollama pull llama3.1`.")
            model = None
        else:
            st.success(f"Ollama connected · {len(models)} model(s)")
            default = next((i for i, m in enumerate(models) if m.startswith("llama3.1:")), 0)
            model = st.selectbox("Model", models, index=default)
        max_revisions = st.slider("Extractor ↔ reviewer revisions", 0, 3, 2,
                                  help="How many times the reviewer may send the extraction back.")
        num_ctx = st.select_slider("Context window (tokens)", [4096, 8192, 16384], value=8192)
        st.divider()
        st.caption("Runs entirely on your machine with Ollama. No API keys, no external services, no cost.")

    st.title("Argument Miner")
    st.markdown(
        "A team of LangChain agents orchestrated with LangGraph that extracts the **thesis, claims and premises** "
        "of a text, how they **support or attack** each other, their **argumentation schemes** and **fallacies**."
    )

    examples = sorted(EXAMPLES.glob("*.txt")) if EXAMPLES.exists() else []
    if examples:
        cols = st.columns(len(examples) + 2)
        cols[0].markdown("**Examples:**")
        for col, path in zip(cols[1:], examples):
            col.button(path.stem.replace("_", " "), on_click=load_example, args=(path,), width="stretch")

    st.text_area("Text to analyse", key="text", height=220,
                 placeholder="Paste an argumentative text: an essay, an opinion piece, a debate turn…")
    left, right = st.columns([3, 1])
    left.file_uploader("…or upload a .txt file", type=["txt", "md"], key="upload", on_change=load_upload)
    text = st.session_state.get("text", "").strip()
    right.write("")
    right.write("")
    clicked = right.button("Analyse", type="primary", icon=":material/play_arrow:", width="stretch",
                           disabled=not (model and text))

    if clicked and model:
        st.session_state.pop("result", None)
        run_analysis(text, model, base_url, num_ctx, max_revisions)

    if "result" in st.session_state:
        st.divider()
        show_result(st.session_state.result)

    with st.expander("How it works"):
        st.graphviz_chart(PIPELINE_DOT, width="stretch")
        st.markdown(
            "Each LLM agent is an LCEL chain with **structured output** (Pydantic). The reviewer runs exact "
            "structural checks in Python before asking the model, and sends problems back to the extractor "
            "(bounded loop). The scheme analyst and the evaluator run **in parallel**."
        )


main()
