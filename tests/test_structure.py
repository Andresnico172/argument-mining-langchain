from argmining.schemas import ExtractionOutput
from argmining.structure import build_arguments, validate_extraction

from .conftest import BAD_EXTRACTION, GOOD_EXTRACTION, unit


def test_well_formed_extraction_has_no_issues(segments):
    assert validate_extraction(segments, GOOD_EXTRACTION) == []


def test_reports_unknown_target_and_missing_units(segments):
    issues = validate_extraction(segments, BAD_EXTRACTION)
    assert any("unknown unit 'A9'" in i for i in issues)
    assert any("without an annotation: A3" in i for i in issues)


def test_reports_cycles_self_loops_and_missing_conclusion(segments):
    extraction = ExtractionOutput(
        units=[
            unit("A1", "premise", "support", "A2"),
            unit("A2", "premise", "support", "A1"),
            unit("A3", "premise", "attack", "A3"),
        ]
    )
    issues = validate_extraction(segments, extraction)
    assert any("No conclusion" in i for i in issues)
    assert any("Circular" in i for i in issues)
    assert any("A3 cannot support or attack itself" in i for i in issues)


def test_major_claim_is_root_and_premises_need_a_target(segments):
    extraction = ExtractionOutput(
        units=[
            unit("A1", "major_claim", "support", "A2"),
            unit("A2", "premise"),
            unit("A3", "premise", "attack", "A1"),
        ]
    )
    issues = validate_extraction(segments, extraction)
    assert any("A1 is the major claim" in i for i in issues)
    assert any("A2 is a premise" in i for i in issues)


def test_duplicate_annotations(segments):
    extraction = GOOD_EXTRACTION.model_copy(update={"units": [*GOOD_EXTRACTION.units, unit("A2", "claim", "support", "A1")]})
    assert any("annotated more than once" in i for i in validate_extraction(segments, extraction))


def test_build_arguments_groups_support_and_attack(segments):
    [arg] = build_arguments(segments, GOOD_EXTRACTION)
    assert (arg.id, arg.conclusion, arg.premises, arg.attackers) == ("ARG1", "A1", ["A2"], ["A3"])


def test_build_arguments_ignores_invalid_relations(segments):
    assert build_arguments(segments, BAD_EXTRACTION) == []
