from __future__ import annotations

from collections import defaultdict

import pytest
from langchain_core.runnables import RunnableLambda

from argmining.schemas import (
    ArgumentScore,
    EvaluationOutput,
    ExtractionOutput,
    ReviewOutput,
    SchemeAnalysis,
    SchemesOutput,
    Segment,
    SegmentationOutput,
    UnitAnnotation,
    UnitCheck,
)


class ScriptedLLM:
    """Stands in for a chat model: returns queued answers per output schema.

    It records the prompt each agent received so tests can check what was sent.
    """

    def __init__(self, answers: dict[type, list]):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.prompts: dict[type, list[str]] = defaultdict(list)

    def with_structured_output(self, schema, **_):
        def respond(prompt_value):
            self.prompts[schema].append(prompt_value.to_string())
            return self.answers[schema].pop(0)

        return RunnableLambda(respond)


SEGMENTS = [
    Segment(id="A1", text="Cities should ban cars from their centres."),
    Segment(id="A2", text="Traffic is the main source of urban air pollution."),
    Segment(id="A3", text="Some shops fear losing customers."),
]

def unit(id, role, relation=None, target=None):
    return UnitAnnotation(id=id, role=role, relation=relation, target=target)


GOOD_EXTRACTION = ExtractionOutput(
    units=[
        unit("A1", "major_claim"),
        unit("A2", "premise", "support", "A1"),
        unit("A3", "premise", "attack", "A1"),
    ]
)

BAD_EXTRACTION = ExtractionOutput(
    units=[unit("A1", "major_claim"), unit("A2", "premise", "support", "A9")]
)


@pytest.fixture
def segments():
    return [s.model_copy() for s in SEGMENTS]


def happy_answers(extractions=(GOOD_EXTRACTION,), reviews=(ReviewOutput(issues=[], approved=True),)):
    return {
        SegmentationOutput: [SegmentationOutput(language="English", units=[s.text for s in SEGMENTS])],
        ExtractionOutput: list(extractions),
        ReviewOutput: list(reviews),
        SchemesOutput: [
            SchemesOutput(
                arguments=[
                    SchemeAnalysis(
                        argument_id="ARG1",
                        scheme="cause_to_effect",
                        implicit_premises=["Less traffic means less pollution."],
                        critical_questions=["Are there other major pollution sources?"],
                    ),
                    SchemeAnalysis(argument_id="ARG99", scheme="other", implicit_premises=[], critical_questions=[]),
                ]
            )
        ],
        EvaluationOutput: [
            EvaluationOutput(
                unit_checks=[
                    UnitCheck(unit_id="A1", reasoning="A plain thesis.", fallacy="none"),
                    UnitCheck(unit_id="A2", reasoning="No data given.", fallacy="hasty_generalization"),
                ],
                scores=[ArgumentScore(argument_id="ARG1", strength=3, rationale="Plausible but unsupported.")],
                overall="Reasonable thesis with thin evidence.",
            )
        ],
    }
