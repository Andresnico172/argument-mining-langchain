"""Wiring of the agents into a LangGraph state machine.

    segmenter -> extractor -> reviewer --(issues, budget left)--> extractor
                                       `--(approved)--> scheme_analyst + evaluator (parallel)
                                                          -> synthesizer
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import ArgumentationAgents, PipelineState, synthesize


def build_graph(llm: BaseChatModel, max_revisions: int = 2) -> CompiledStateGraph:
    """Build the pipeline for any chat model that supports structured output."""
    agents = ArgumentationAgents(llm)

    def after_segmenter(state: PipelineState) -> str:
        return "extractor" if state["segments"] else "synthesizer"

    def after_reviewer(state: PipelineState) -> str | list[str]:
        if state["feedback"] and state["revisions"] < max_revisions:
            return "extractor"
        # Accepted, or out of revision budget: fan out to both analysts at once.
        return ["scheme_analyst", "evaluator"]

    graph = StateGraph(PipelineState)
    graph.add_node("segmenter", agents.segment)
    graph.add_node("extractor", agents.extract)
    graph.add_node("reviewer", agents.review)
    graph.add_node("scheme_analyst", agents.analyze_schemes)
    graph.add_node("evaluator", agents.evaluate)
    graph.add_node("synthesizer", synthesize)

    graph.add_edge(START, "segmenter")
    graph.add_conditional_edges("segmenter", after_segmenter, ["extractor", "synthesizer"])
    graph.add_edge("extractor", "reviewer")
    graph.add_conditional_edges("reviewer", after_reviewer, ["extractor", "scheme_analyst", "evaluator"])
    graph.add_edge(["scheme_analyst", "evaluator"], "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()
