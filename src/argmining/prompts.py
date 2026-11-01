"""Prompt templates, one per agent.

Instructions are in English (small local models follow them more reliably),
but every free-text field is written in the language of the input text.
"""

from langchain_core.prompts import ChatPromptTemplate

LANGUAGE_RULE = (
    "Write every free-text field (issues, premises, questions, explanations, "
    "rationales, assessments) in {language}. Ids, role, relation, scheme and "
    "fallacy names stay exactly as defined."
)

SEGMENTER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in argument mining. Split the text into argumentative "
            "discourse units (ADUs): minimal spans that state a position, a reason, "
            "evidence or an objection. Rules:\n"
            "- Copy each unit verbatim from the text; do not paraphrase or merge ideas.\n"
            "- A sentence with two separate reasons gives two units.\n"
            "- Leave out greetings, rhetorical filler and purely descriptive background.\n"
            "- Keep the original reading order.",
        ),
        ("human", "TEXT:\n{text}"),
    ]
)

EXTRACTOR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in argumentation theory. You receive the argumentative "
            "units of a text, each with an id. Annotate every unit once with:\n"
            "- role:\n"
            "  - major_claim: the central thesis the author defends. Exactly one.\n"
            "  - claim: an intermediate conclusion, or an objection the author "
            "raises, that bears directly on the major claim.\n"
            "  - premise: a reason, piece of evidence or example for or against a claim.\n"
            "- target: the id of the unit THIS unit is a reason for or against. "
            "Reasons point towards the conclusion they back, never the other way "
            "round. null for the major claim.\n"
            "- relation: 'support' if this unit is a reason FOR its target, 'attack' "
            "if it is a reason AGAINST it (an objection or a rebuttal). null for the "
            "major claim.\n\n"
            "Example. Units:\n"
            "A1: We should plant more trees in the city.\n"
            "A2: Trees lower summer temperatures.\n"
            "A3: Shade reduces the need for air conditioning.\n"
            "A4: Some say maintenance is too expensive.\n"
            "A5: But the energy savings outweigh that cost.\n"
            "Annotation:\n"
            "A1 major_claim (no target)\n"
            "A2 claim, support -> A1\n"
            "A3 premise, support -> A2\n"
            "A4 claim, attack -> A1\n"
            "A5 premise, attack -> A4\n\n"
            "Only use the given ids.",
        ),
        ("human", "FULL TEXT:\n{text}\n\nUNITS:\n{units}{feedback}"),
    ]
)

REVIEWER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict reviewer of argument annotations. Check the proposed "
            "structure against the original text:\n"
            "- Is the major claim really the author's central thesis?\n"
            "- Does each relation reflect what the text says (support vs attack)?\n"
            "- Is any premise attached to the wrong conclusion?\n"
            "Reject ONLY for these errors: wrong major claim, a unit pointing to the "
            "wrong target, or support and attack swapped. Whether a unit is labelled "
            "claim or premise is a judgement call and never a reason to reject. "
            "If none of those errors is present, approve with an empty issue list. "
            "When you reject, list concrete issues that cite unit ids. " + LANGUAGE_RULE,
        ),
        ("human", "FULL TEXT:\n{text}\n\nPROPOSED STRUCTURE:\n{structure}"),
    ]
)

SCHEME_ANALYST = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in Walton's argumentation schemes. For each argument "
            "(a conclusion with its supporting premises), identify:\n"
            "- scheme: the inference pattern linking premises to conclusion "
            "(expert_opinion, example, analogy, cause_to_effect, consequences, sign, "
            "popular_opinion, practical_reasoning, definition, other).\n"
            "- implicit_premises: unstated assumptions (enthymemes) the inference "
            "needs. Empty list if none.\n"
            "- critical_questions: the scheme's critical questions that the text "
            "leaves open, phrased as questions.\n"
            "Return one entry per argument id. " + LANGUAGE_RULE,
        ),
        ("human", "FULL TEXT:\n{text}\n\nARGUMENTS:\n{arguments}"),
    ]
)

EVALUATOR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a critical-thinking expert who evaluates arguments.\n"
            "- unit_checks: go through EVERY unit, one entry each. First reason in "
            "one sentence whether it matches one of these definitions, then give the "
            "fallacy name, or 'none':\n"
            "  ad_hominem: attacks the people holding a view instead of the view.\n"
            "  straw_man: refutes a distorted version of the opposing position.\n"
            "  false_dilemma: presents only two options when more exist.\n"
            "  hasty_generalization: a general conclusion from too few cases.\n"
            "  slippery_slope: claims a chain of unavoidable bad consequences.\n"
            "  appeal_to_authority: relies on an unnamed or irrelevant authority.\n"
            "  appeal_to_popularity: true because many people believe it.\n"
            "  post_hoc: causation inferred from mere sequence in time.\n"
            "  circular_reasoning: the conclusion is assumed in a premise.\n"
            "  red_herring: an irrelevant point that diverts from the issue.\n"
            "  Most units are not fallacious: use 'none' unless the definition clearly fits.\n"
            "- scores: rate each argument from 1 (very weak) to 5 (very strong) by "
            "relevance, sufficiency and acceptability of its premises.\n"
            "- overall: two or three sentences on the argumentation as a whole.\n"
            + LANGUAGE_RULE,
        ),
        ("human", "FULL TEXT:\n{text}\n\nUNITS:\n{units}\n\nARGUMENTS:\n{arguments}"),
    ]
)
