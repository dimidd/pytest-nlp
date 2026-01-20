"""Tests for examples in the Medical README.

These tests verify that all code examples in MEDICAL_README.md work correctly.
Code blocks are extracted into the README using markdown-autodocs.
"""

import re

import pytest

# =============================================================================
# Medical Quick Start Example
# =============================================================================


@pytest.fixture
def requires_medical():
    """Skip if medspacy is not installed."""
    pytest.importorskip("medspacy")


def test_medical_quick_start(requires_medical) -> None:
    """Medical quick start example."""
    from pytest_nlp import assert_has_drug, assert_semantic_contains

    note = """Patient was prescribed Ibuprofen on 01/12/2023 for headaches.
    Patient reported mild side effects. Ibuprofen was discontinued on 12/30/2023."""

    # Semantic: check if discontinuation is mentioned
    assert_semantic_contains(
        query="Ibuprofen was discontinued",
        document=note,
        threshold=0.7,
    )

    # Medical NER: verify drug is mentioned
    assert_has_drug(note, drug_name="ibuprofen")


# =============================================================================
# Medical Semantic Similarity Examples
# =============================================================================


def test_medical_semantic_similarity(requires_medical) -> None:
    """Test semantic similarity for clinical text."""
    from pytest_nlp import assert_semantic_contains, semantic_contains

    # Check if clinical concept is semantically present
    is_contained, score = semantic_contains(
        query="drug discontinued",
        document="Ibuprofen was discontinued on 12/30/2023.",
        threshold=0.5,
    )
    assert is_contained

    # Verify diagnosis mentioned
    assert_semantic_contains(
        query="patient has high blood pressure",
        document="Assessment: Hypertension, well-controlled on current medications.",
        threshold=0.4,
    )


def test_medical_sentence_match(requires_medical) -> None:
    """Test SentenceMatch for clinical sections."""
    from pytest_nlp import SentenceMatch, assert_semantic_contains

    discharge_summary = """Chief Complaint: Chest pain for 2 days.
    History: Patient is a 65-year-old male with history of diabetes.
    Assessment: Acute coronary syndrome ruled out.
    Plan: Continue current medications. Follow up in 2 weeks."""

    # Check the Plan section
    assert_semantic_contains(
        query="maintain current treatment",
        document=discharge_summary,
        sentences=SentenceMatch("Plan", mode="first"),
        threshold=0.4,
    )


def test_medical_drug_sentence_match(requires_medical) -> None:
    """Test finding drug mentions with SentenceMatch."""
    from pytest_nlp import SentenceMatch, assert_semantic_contains

    note = """Patient was prescribed Ibuprofen 200mg on 01/12/2023.
    Patient reported mild side effects including nausea.
    Dosage increased to 400mg on 02/15/2023.
    Ibuprofen was discontinued on 12/30/2023 due to GI symptoms."""

    # Find first sentence about Ibuprofen
    assert_semantic_contains(
        query="prescribed Ibuprofen",
        document=note,
        sentences=SentenceMatch("ibuprofen", mode="first"),
        threshold=0.5,
    )

    # Find last sentence about Ibuprofen (likely discontinuation)
    assert_semantic_contains(
        query="Ibuprofen discontinued",
        document=note,
        sentences=SentenceMatch("ibuprofen", mode="last"),
        threshold=0.5,
    )


# =============================================================================
# Medical SpaCy Matchers Examples
# =============================================================================


def test_medical_token_matching(requires_medical) -> None:
    """Test token matching for clinical text."""
    from pytest_nlp import assert_matches_tokens, match_tokens

    # Match medication administration patterns
    assert_matches_tokens(
        doc="Patient was prescribed Metformin 500mg twice daily.",
        patterns=[[{"LEMMA": "prescribe"}]],
    )

    # Match symptom patterns
    results = match_tokens(
        doc="Patient reports severe chest pain radiating to left arm.",
        patterns=[[{"LOWER": "chest"}, {"LOWER": "pain"}]],
    )
    assert len(results) >= 1


def test_medical_phrase_matching(requires_medical) -> None:
    """Test phrase matching for clinical text."""
    from pytest_nlp import assert_matches_phrases

    # Match clinical phrases
    assert_matches_phrases(
        doc="Patient presents with shortness of breath and chest pain.",
        phrases=["shortness of breath", "chest pain"],
    )

    # Case-insensitive drug matching
    assert_matches_phrases(
        doc="Started METFORMIN 500mg BID",
        phrases=["metformin"],
        attr="LOWER",
    )


def test_medical_dependency_matching(requires_medical) -> None:
    """Test dependency matching for clinical relationships."""
    from pytest_nlp import assert_matches_dependency

    # Match drug discontinuation pattern
    assert_matches_dependency(
        doc="Ibuprofen was discontinued on 12/30/2023.",
        patterns=[
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "discontinue"}},
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "drug",
                "RIGHT_ATTRS": {"LOWER": "ibuprofen", "DEP": "nsubjpass"},
            },
        ],
    )


# =============================================================================
# Medical Constituency Parsing Examples
# =============================================================================


def test_medical_constituency_parsing(requires_medical) -> None:
    """Test constituency parsing for clinical text."""
    from pytest_nlp import get_constituency_tree, match_constituency

    # Get constituency tree for clinical sentence
    trees = get_constituency_tree("Ibuprofen was discontinued.")
    assert len(trees) == 1
    assert "discontinued" in trees[0]

    # Match drug discontinuation with variable capture
    # Use partial matching (...) at S level to account for punctuation
    matches = match_constituency(
        doc="Ibuprofen was discontinued on 12/30/2023.",
        pattern="(S (NP (NNP ?drug)) ...)",
    )
    assert len(matches) >= 1
    assert matches[0]["drug"] == "Ibuprofen"


# =============================================================================
# Medical NER Examples
# =============================================================================


def test_medical_entity_extraction(requires_medical) -> None:
    """Test medical entity extraction."""
    from pytest_nlp import MedicalEntity, extract_medical_entities

    note = """Patient takes ibuprofen for headaches. No fever reported.
    Possible pneumonia. History of diabetes."""

    entities = extract_medical_entities(note)
    assert len(entities) > 0

    # Get only drugs
    drugs = extract_medical_entities(note, labels=["DRUG"])
    assert any(e.text.lower() == "ibuprofen" for e in drugs)


def test_medical_assert_has_drug(requires_medical) -> None:
    """Test drug entity assertions."""
    from pytest_nlp import assert_has_drug

    # Assert any drug is mentioned
    assert_has_drug("Patient takes ibuprofen daily.")

    # Assert specific drug
    assert_has_drug("Patient takes ibuprofen daily.", drug_name="ibuprofen")

    # Assert drug is NOT negated (affirmed)
    assert_has_drug("Patient takes ibuprofen daily.", negated=False)


def test_medical_assert_has_problem(requires_medical) -> None:
    """Test problem entity assertions."""
    from pytest_nlp import assert_has_problem

    # Assert any problem is mentioned
    assert_has_problem("Patient presents with fever.")

    # Assert specific problem
    assert_has_problem("Patient presents with fever.", problem="fever")


def test_medical_entity_negation(requires_medical) -> None:
    """Test medical entity negation detection."""
    from pytest_nlp import assert_entity_negated, assert_entity_not_negated

    note = "Patient denies chest pain. Reports headache."

    # Assert chest pain is negated
    assert_entity_negated(note, entity_text="chest pain")

    # Assert headache is NOT negated
    assert_entity_not_negated(note, entity_text="headache")


def test_medical_context_modifiers(requires_medical) -> None:
    """Test context modifier detection."""
    from pytest_nlp import extract_medical_entities

    note = """
    Patient denies chest pain.
    No fever reported.
    Possible pneumonia.
    History of diabetes.
    """

    entities = extract_medical_entities(note)

    # Check negation
    negated = [e for e in entities if e.is_negated]
    assert len(negated) >= 1


# =============================================================================
# Medical AMR Examples (requires amrlib)
# =============================================================================


@pytest.fixture
def requires_amr():
    """Skip if amrlib is not installed."""
    pytest.importorskip("amrlib")


def test_medical_amr_parsing(requires_medical, requires_amr) -> None:
    """Test AMR parsing for clinical sentences."""
    from pytest_nlp import assert_has_concept, parse_amr

    # Parse clinical action
    graph = parse_amr("The doctor discontinued the medication.")
    assert graph.has_concept("discontinue-01")
    assert graph.has_concept("doctor")

    # Assert discontinuation concept exists
    assert_has_concept("The doctor discontinued Ibuprofen.", "discontinue-01")


def test_medical_amr_negation(requires_medical, requires_amr) -> None:
    """Test AMR negation detection for clinical text."""
    from pytest_nlp import assert_is_negated, assert_not_negated

    # Negated clinical findings (using explicit negation)
    assert_is_negated("The doctor did not discontinue the medication.")
    assert_not_negated("The doctor discontinued the medication.")


def test_medical_amr_similarity(requires_medical, requires_amr) -> None:
    """Test AMR similarity for clinical sentences."""
    from pytest_nlp import assert_amr_similarity, sentence_amr_similarity

    # Compare semantically similar clinical statements
    score = sentence_amr_similarity(
        "The doctor discontinued the medication.",
        "The physician stopped the drug.",
    )
    assert score > 0

