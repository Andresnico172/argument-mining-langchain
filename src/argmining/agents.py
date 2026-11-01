"""The LLM agents (graph nodes) and the shared state they read and write."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from . import prompts
from .schemas import (
    Argument,
    ArgumentMap,
    EvaluationOutput,
    ExtractionOutput,
    ReviewOutput,
    SchemesOutput,
    Segment,
    SegmentationOutput,
)
from .structure import build_arguments, validate_extraction


class PipelineState(TypedDict, total=False):
    text: str
    language: str
    segments: list[Segment]
    extraction: ExtractionOutput
    feedback: list[str]
    revisions: int
    arguments: list[Argument]
    schemes: SchemesOutput
    evaluation: EvaluationOutput
    result: ArgumentMap


# --- Rendering state as prompt text ------------------------------------------


def render_units(segments: list[Segment]) -> str:
    return "\n".join(f"{s.id}: {s.text}" for s in segments)


def render_structure(segments: list[Segment], extraction: ExtractionOutput) -> str:
    annotations = {u.id: u for u in extraction.units}
    lines = []
    for s in segments:
        u = annotations.get(s.id)
        if u is None:
            lines.append(f"{s.id} [no annotation]: {s.text}")
        elif u.target:
            lines.append(f"{s.id} [{u.role}] --{u.relation}--> {u.target}: {s.text}")
        else:
            lines.append(f"{s.id} [{u.role}]: {s.text}")
    return "\n".join(lines)


def render_arguments(arguments: list[Argument], segments: list[Segment], roles: dict[str, str]) -> str:
    text = {s.id: s.text for s in segments}
    blocks = []
    for arg in arguments:
        lines = [f'{arg.id}: conclusion {arg.conclusion} ({roles.get(arg.conclusion)}): "{text[arg.conclusion]}"']
        lines += [f'  supported by {p}: "{text[p]}"' for p in arg.premises]
        lines += [f'  attacked by {a}: "{text[a]}"' for a in arg.attackers]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# --- Agents ------------------------------------------------------------------


class ArgumentationAgents:
    """Each method is a LangGraph node: it takes the state and returns an update."""

    def __init__(self, llm: BaseChatModel, retries: int = 3):
        self.llm = llm
        self.retries = retries
        self.segmenter = self._chain("segmenter", prompts.SEGMENTER, SegmentationOutput)
        self.extractor = self._chain("extractor", prompts.EXTRACTOR, ExtractionOutput)
        self.reviewer = self._chain("reviewer", prompts.REVIEWER, ReviewOutput)
        self.scheme_analyst = self._chain("scheme_analyst", prompts.SCHEME_ANALYST, SchemesOutput)
        self.evaluator = self._chain("evaluator", prompts.EVALUATOR, EvaluationOutput)

    def _chain(self, name: str, prompt: ChatPromptTemplate, schema: type[BaseModel]) -> Runnable:
        # LCEL: prompt -> model constrained to the schema. Small models sometimes
        # emit JSON that fails validation, so the whole chain is retried.
        chain = prompt | self.llm.with_structured_output(schema)
        return chain.with_retry(stop_after_attempt=self.retries).with_config(run_name=name)

    def segment(self, state: PipelineState) -> PipelineState:
        out: SegmentationOutput = self.segmenter.invoke({"text": state["text"]})
        units = [u.strip() for u in out.units if u.strip()]
        segments = [Segment(id=f"A{i}", text=u) for i, u in enumerate(units, start=1)]
        return {"segments": segments, "language": out.language or "English", "feedback": [], "revisions": 0}

    def extract(self, state: PipelineState) -> PipelineState:
        feedback = state.get("feedback") or []
        revisions = state.get("revisions", 0)
        note = ""
        if feedback:
            previous = render_structure(state["segments"], state["extraction"])
            issues = "\n".join(f"- {i}" for i in feedback)
            note = (
                f"\n\nYOUR PREVIOUS ANSWER:\n{previous}\n\n"
                f"A REVIEWER FOUND THESE PROBLEMS. FIX THEM:\n{issues}"
            )
            revisions += 1
        out: ExtractionOutput = self.extractor.invoke(
            self._inputs(state, units=render_units(state["segments"]), feedback=note)
        )
        return {"extraction": out, "revisions": revisions}

    def review(self, state: PipelineState) -> PipelineState:
        segments, extraction = state["segments"], state["extraction"]
        # Exact checks first; only ask the model when the structure is well formed.
        issues = validate_extraction(segments, extraction)
        if not issues:
            verdict: ReviewOutput = self.reviewer.invoke(
                self._inputs(state, structure=render_structure(segments, extraction))
            )
            if not verdict.approved:
                issues = verdict.issues or ["The reviewer rejected the structure without details."]
        return {"feedback": issues, "arguments": build_arguments(segments, extraction)}

    def analyze_schemes(self, state: PipelineState) -> PipelineState:
        arguments = state["arguments"]
        if not arguments:
            return {"schemes": SchemesOutput(arguments=[])}
        out: SchemesOutput = self.scheme_analyst.invoke(
            self._inputs(state, arguments=self._arguments_text(state))
        )
        known = {a.id for a in arguments}
        return {"schemes": SchemesOutput(arguments=[s for s in out.arguments if s.argument_id in known])}

    def evaluate(self, state: PipelineState) -> PipelineState:
        arguments = state["arguments"]
        if not arguments:
            return {"evaluation": EvaluationOutput(unit_checks=[], scores=[], overall="")}
        out: EvaluationOutput = self.evaluator.invoke(
            self._inputs(state, units=render_units(state["segments"]), arguments=self._arguments_text(state))
        )
        known = {a.id for a in arguments}
        out.scores = [s for s in out.scores if s.argument_id in known]
        return {"evaluation": out}

    @staticmethod
    def _inputs(state: PipelineState, **extra: str) -> dict[str, str]:
        return {"text": state["text"], "language": state.get("language", "English"), **extra}

    @staticmethod
    def _arguments_text(state: PipelineState) -> str:
        roles = state["extraction"].role_map
        return render_arguments(state["arguments"], state["segments"], roles)


def synthesize(state: PipelineState) -> PipelineState:
    """Deterministic node: assemble the final argument map from the state."""
    extraction = state.get("extraction") or ExtractionOutput(units=[])
    segments = state.get("segments", [])
    known = {s.id for s in segments}
    result = ArgumentMap(
        segments=segments,
        roles={k: v for k, v in extraction.role_map.items() if k in known},
        relations=[r for r in extraction.relations if r.source in known and r.target in known],
        arguments=state.get("arguments", []),
        schemes=state["schemes"].arguments if "schemes" in state else [],
        evaluation=state.get("evaluation"),
        revisions=state.get("revisions", 0),
        unresolved_issues=state.get("feedback", []),
    )
    return {"result": result}
