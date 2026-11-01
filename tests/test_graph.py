import os

import httpx
import pytest

from argmining.export import to_dot, to_markdown, to_mermaid, to_networkx
from argmining.graph import build_graph
from argmining.schemas import ExtractionOutput, ReviewOutput, SegmentationOutput

from .conftest import BAD_EXTRACTION, GOOD_EXTRACTION, ScriptedLLM, happy_answers


def test_happy_path_runs_every_agent_once():
    llm = ScriptedLLM(happy_answers())
    result = build_graph(llm).invoke({"text": "..."})["result"]

    assert [s.id for s in result.conclusions] == ["A1"]
    assert result.revisions == 0 and result.unresolved_issues == []
    assert [s.argument_id for s in result.schemes] == ["ARG1"]  # unknown ARG99 dropped
    assert result.evaluation.scores[0].strength == 3
    assert all(not queue for queue in llm.answers.values())


def test_structural_issues_loop_back_without_calling_reviewer():
    llm = ScriptedLLM(happy_answers(extractions=[BAD_EXTRACTION, GOOD_EXTRACTION]))
    result = build_graph(llm).invoke({"text": "..."})["result"]

    assert result.revisions == 1
    assert len(llm.prompts[ReviewOutput]) == 1  # only the second, well-formed attempt
    second_prompt = llm.prompts[ExtractionOutput][1]
    assert "A REVIEWER FOUND THESE PROBLEMS" in second_prompt and "A9" in second_prompt


def test_reviewer_rejection_feeds_back_into_extractor():
    reviews = [ReviewOutput(issues=["A3 attacks A1, it does not support it."], approved=False),
               ReviewOutput(issues=[], approved=True)]
    llm = ScriptedLLM(happy_answers(extractions=[GOOD_EXTRACTION, GOOD_EXTRACTION], reviews=reviews))
    result = build_graph(llm).invoke({"text": "..."})["result"]

    assert result.revisions == 1
    assert "A3 attacks A1" in llm.prompts[ExtractionOutput][1]


def test_revision_budget_is_respected():
    llm = ScriptedLLM(happy_answers(extractions=[BAD_EXTRACTION] * 3))
    result = build_graph(llm, max_revisions=2).invoke({"text": "..."})["result"]

    assert result.revisions == 2
    assert result.unresolved_issues  # reported, not hidden
    assert len(llm.prompts[ExtractionOutput]) == 3


def test_empty_text_short_circuits():
    llm = ScriptedLLM({SegmentationOutput: [SegmentationOutput(language="English", units=["  "])]})
    result = build_graph(llm).invoke({"text": "Hello."})["result"]
    assert result.segments == [] and result.arguments == []


def test_exports():
    result = build_graph(ScriptedLLM(happy_answers())).invoke({"text": "..."})["result"]
    graph = to_networkx(result)
    assert graph.edges["A3", "A1"]["type"] == "attack"
    mermaid = to_mermaid(result)
    assert "A2 -->|support| A1" in mermaid and "A3 -.->|attack| A1" in mermaid
    dot = to_dot(result)
    assert dot.startswith("digraph") and "A3 -> A1" in dot and "style=dashed" in dot
    report = to_markdown(result)
    assert "## Conclusions" in report and "`cause_to_effect`" in report and "hasty_generalization" in report


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/version", timeout=1).is_success
    except httpx.HTTPError:
        return False


@pytest.mark.ollama
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1" or not _ollama_up(), reason="needs Ollama and RUN_OLLAMA_TESTS=1")
def test_end_to_end_with_ollama():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=os.getenv("ARGMINING_MODEL", "llama3.1"), temperature=0, num_ctx=8192)
    text = (
        "Schools should start later in the morning. Teenagers' sleep cycles shift later during "
        "puberty, so early classes leave them chronically tired. Studies show that later start "
        "times improve attendance. Critics argue that it complicates bus schedules."
    )
    result = build_graph(llm).invoke({"text": text})["result"]
    assert result.segments and result.conclusions
