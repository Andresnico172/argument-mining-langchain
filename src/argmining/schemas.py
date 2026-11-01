"""Typed contracts exchanged between the agents of the pipeline.

The ``*Output`` models are what each LLM agent must return (they are turned
into a JSON schema and enforced with structured output). The rest are domain
objects assembled by plain Python code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["major_claim", "claim", "premise"]
RelationType = Literal["support", "attack"]

# Argumentation schemes after Walton, Reed & Macagno (2008), reduced to the
# ones that show up most often in essays and opinion pieces.
Scheme = Literal[
    "expert_opinion",
    "example",
    "analogy",
    "cause_to_effect",
    "consequences",
    "sign",
    "popular_opinion",
    "practical_reasoning",
    "definition",
    "other",
]

FallacyName = Literal[
    "ad_hominem",
    "straw_man",
    "false_dilemma",
    "hasty_generalization",
    "slippery_slope",
    "appeal_to_authority",
    "appeal_to_popularity",
    "post_hoc",
    "circular_reasoning",
    "red_herring",
    "other",
]


# --- LLM outputs ------------------------------------------------------------


class SegmentationOutput(BaseModel):
    """Argumentative discourse units (ADUs) found in the text."""

    language: str = Field(description="Language the text is written in, in English, e.g. 'Spanish'.")
    units: list[str] = Field(
        description="Argumentative discourse units copied verbatim from the text, in reading order."
    )


class Relation(BaseModel):
    source: str
    target: str
    type: RelationType


class UnitAnnotation(BaseModel):
    """One unit, its role and the single unit it points to (a tree, as in the AAE corpus).

    Asking "what does this unit support?" per unit avoids the direction mix-ups
    small models make when they write free-form source -> target edges.
    """

    id: str = Field(description="Id of the unit, e.g. 'A3'.")
    role: Role
    relation: RelationType | None = Field(
        description="Whether this unit supports or attacks its target. null only for the major claim."
    )
    target: str | None = Field(
        description="Id of the unit this one supports or attacks. null only for the major claim."
    )


class ExtractionOutput(BaseModel):
    units: list[UnitAnnotation]

    @property
    def role_map(self) -> dict[str, Role]:
        return {u.id: u.role for u in self.units}

    @property
    def relations(self) -> list[Relation]:
        return [
            Relation(source=u.id, target=u.target, type=u.relation)
            for u in self.units
            if u.target and u.relation
        ]


class ReviewOutput(BaseModel):
    issues: list[str] = Field(
        description="Concrete problems in the extraction. Empty list if it is correct."
    )
    approved: bool


class SchemeAnalysis(BaseModel):
    argument_id: str
    scheme: Scheme
    implicit_premises: list[str] = Field(
        description="Unstated premises the inference needs in order to hold."
    )
    critical_questions: list[str] = Field(
        description="Critical questions of the scheme that the text leaves unanswered."
    )


class SchemesOutput(BaseModel):
    arguments: list[SchemeAnalysis]


class Fallacy(BaseModel):
    name: FallacyName
    unit_ids: list[str]
    explanation: str


class ArgumentScore(BaseModel):
    argument_id: str
    strength: Literal[1, 2, 3, 4, 5] = Field(description="1 = very weak, 5 = very strong.")
    rationale: str


class UnitCheck(BaseModel):
    """Per-unit fallacy check. The reasoning comes before the label on purpose:
    the model writes its justification first and then commits to a verdict."""

    unit_id: str
    reasoning: str = Field(description="One sentence: does this unit match any fallacy definition, and why?")
    fallacy: FallacyName | Literal["none"]


class EvaluationOutput(BaseModel):
    unit_checks: list[UnitCheck] = Field(description="One entry per unit, in order.")
    scores: list[ArgumentScore]
    overall: str = Field(
        description="Two or three sentences on the overall quality of the argumentation."
    )

    @property
    def fallacies(self) -> list[Fallacy]:
        return [
            Fallacy(name=c.fallacy, unit_ids=[c.unit_id], explanation=c.reasoning)
            for c in self.unit_checks
            if c.fallacy != "none"
        ]


# --- Domain objects built by code -------------------------------------------


class Segment(BaseModel):
    id: str
    text: str


class Argument(BaseModel):
    """A conclusion together with the units that directly support or attack it."""

    id: str
    conclusion: str
    premises: list[str]
    attackers: list[str]


class ArgumentMap(BaseModel):
    """Final result of the pipeline."""

    segments: list[Segment]
    roles: dict[str, Role]
    relations: list[Relation]
    arguments: list[Argument]
    schemes: list[SchemeAnalysis]
    evaluation: EvaluationOutput | None
    revisions: int
    unresolved_issues: list[str]

    def text_of(self, unit_id: str) -> str:
        return next((s.text for s in self.segments if s.id == unit_id), "")

    @property
    def conclusions(self) -> list[Segment]:
        """Major claims first, then intermediate claims."""
        order = {"major_claim": 0, "claim": 1}
        found = [s for s in self.segments if self.roles.get(s.id) in order]
        return sorted(found, key=lambda s: order[self.roles[s.id]])
