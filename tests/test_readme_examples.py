"""Tests for examples in the main README.

These tests verify that all code examples in README.md work correctly.
Code blocks are extracted into the README using markdown-autodocs.
"""

import re

import pytest

from pytest_nlp import (
    SentenceMatch,
    assert_matches_constituency,
    assert_matches_dependency,
    assert_matches_phrases,
    assert_matches_tokens,
    assert_semantic_contains,
    get_constituency_tree,
    match_constituency,
    match_phrases,
    match_tokens,
    semantic_contains,
    semantic_similarity,
)


# =============================================================================
# Quick Start Example
# =============================================================================


def test_quick_start() -> None:
    """Quick start example for README."""
    from pytest_nlp import assert_semantic_contains

    doc = """The new smartphone features a 6.5-inch display.
    It has an excellent camera system. Battery life is outstanding."""

    # Semantic matching: check a sentiment of an aspect of the document
    assert_semantic_contains(
        query="great battery performance",
        document=doc,
        threshold=0.6,
    )

    # Check only last sentence
    assert_semantic_contains(
        query="long battery life",
        document=doc,
        sentences=[-1],
        threshold=0.5,
    )


# =============================================================================
# Semantic Similarity Examples
# =============================================================================


def test_semantic_similarity_basic() -> None:
    """Test basic semantic similarity."""
    from pytest_nlp import semantic_similarity

    # Get similarity score between two texts
    score = semantic_similarity("Hello world", "Hi there world")
    assert score > 0.5


def test_semantic_contains_basic() -> None:
    """Test semantic contains function."""
    from pytest_nlp import semantic_contains

    # Check if query is semantically contained in document
    is_contained, score = semantic_contains(
        query="The project was completed",
        document="We finished the project on Friday.",
        threshold=0.6,
    )
    assert is_contained
    assert score >= 0.6


# =============================================================================
# Sentence Filtering Examples
# =============================================================================


def test_sentence_filtering() -> None:
    """Test sentence filtering options."""
    from pytest_nlp import assert_semantic_contains

    doc = "First sentence. Second sentence. Third sentence. Last sentence."

    # Check only last sentence
    assert_semantic_contains("Last sentence", doc, sentences=[-1], threshold=0.5)

    # Check first two sentences
    assert_semantic_contains("First", doc, sentences=[0, 1], threshold=0.3)

    # Check last 3 sentences using slice
    assert_semantic_contains("Last", doc, sentences=slice(-3, None), threshold=0.3)


# =============================================================================
# Pattern-Based Sentence Selection Examples
# =============================================================================


def test_sentence_match_example() -> None:
    """Test SentenceMatch for pattern-based sentence selection."""
    import re

    from pytest_nlp import SentenceMatch, assert_semantic_contains

    doc = """The order was placed on Monday.
    Processing took 2 business days.
    Shipping cost was $15.99.
    The order arrived on Thursday."""

    # Select first sentence containing "order" (case-insensitive)
    assert_semantic_contains(
        query="order was placed",
        document=doc,
        sentences=SentenceMatch("order", mode="first"),
        threshold=0.5,
    )

    # Select last sentence containing "order"
    assert_semantic_contains(
        query="order arrived",
        document=doc,
        sentences=SentenceMatch("order", mode="last"),
        threshold=0.5,
    )

    # Select all sentences matching a regex pattern (e.g., containing prices)
    assert_semantic_contains(
        query="shipping cost",
        document=doc,
        sentences=SentenceMatch(re.compile(r"\$\d+"), mode="all"),
        threshold=0.3,
    )


# =============================================================================
# Token Matching Examples
# =============================================================================


def test_token_matching() -> None:
    """Test token matching patterns."""
    from pytest_nlp import assert_matches_tokens, match_tokens

    # Match token patterns
    results = match_tokens(
        doc="The quick brown fox jumps over the lazy dog",
        patterns=[[{"LOWER": "quick"}, {"LOWER": "brown"}]],
    )
    assert len(results) >= 1
    assert results[0].text == "quick brown"

    # Assert pattern matches
    assert_matches_tokens(
        doc="The company announced record profits",
        patterns=[[{"LEMMA": "announce"}]],
    )


# =============================================================================
# Phrase Matching Examples
# =============================================================================


def test_phrase_matching() -> None:
    """Test phrase matching."""
    from pytest_nlp import assert_matches_phrases, match_phrases

    # Match exact phrases
    results = match_phrases(
        doc="The quick brown fox",
        phrases=["quick brown fox"],
    )
    assert len(results) >= 1
    assert results[0].text == "quick brown fox"

    # Case-insensitive matching
    assert_matches_phrases(
        doc="Order shipped via FEDEX",
        phrases=["fedex"],
        attr="LOWER",
    )


# =============================================================================
# Dependency Matching Examples
# =============================================================================


def test_dependency_matching() -> None:
    """Test dependency pattern matching."""
    from pytest_nlp import assert_matches_dependency

    # Match dependency patterns (subject-verb-object)
    assert_matches_dependency(
        doc="The engineer fixed the bug yesterday.",
        patterns=[
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "fix"}},
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": "nsubj"},
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {"DEP": "dobj"},
            },
        ],
    )


# =============================================================================
# Constituency Parsing Examples
# =============================================================================


def test_constituency_parsing() -> None:
    """Test constituency parsing."""
    from pytest_nlp import (
        assert_matches_constituency,
        get_constituency_tree,
        match_constituency,
    )

    # Get constituency tree (useful for debugging)
    trees = get_constituency_tree("The cat sat on the mat.")
    assert len(trees) == 1
    assert trees[0].startswith("(")

    # Match with variable capture
    matches = match_constituency(
        doc="The engineer fixed the bug.",
        pattern="(S (NP (DT ?det) (NN ?subject)) ...)",
    )
    assert len(matches) >= 1
    assert matches[0]["subject"] == "engineer"

    # Assert pattern matches
    assert_matches_constituency(
        doc="The cat sat on the mat.",
        pattern="(NP (DT ?det) (NN ?noun))",
    )


# =============================================================================
# AMR Examples (requires amrlib)
# =============================================================================


@pytest.fixture
def requires_amr():
    """Skip if amrlib is not installed."""
    pytest.importorskip("amrlib")


def test_amr_parsing(requires_amr) -> None:
    """Test AMR parsing and concept checking."""
    from pytest_nlp import assert_has_concept, assert_has_role, parse_amr

    # Parse a sentence to AMR
    graph = parse_amr("The boy wants to go.")
    assert graph.has_concept("want-01")
    assert graph.has_concept("boy")

    # Assert concept exists
    assert_has_concept("The manager approved the request.", "approve-01")

    # Assert role exists
    assert_has_role(
        "The boy wants to go.",
        role=":ARG0",
        source_concept="want-01",
    )


def test_amr_negation(requires_amr) -> None:
    """Test AMR negation detection."""
    from pytest_nlp import assert_is_negated, assert_not_negated

    # Check for negation
    assert_is_negated("The boy does not want to go.")
    assert_not_negated("The boy wants to go.")

    # Check specific concept negation
    assert_is_negated(
        "The manager did not approve the request.",
        concept="approve-01",
    )


def test_amr_pattern_matching(requires_amr) -> None:
    """Test AMR pattern matching."""
    from pytest_nlp import assert_amr_pattern, find_concepts, parse_amr

    graph = parse_amr("The boy wants to go.")

    # Find all instances of a concept
    matches = find_concepts(graph, "want-*")  # Wildcard support
    assert len(matches) >= 1

    # Match patterns with specific roles
    assert_amr_pattern(
        "The boy wants to go.",
        concept="want-01",
        roles={":ARG0": "boy", ":ARG1": "*"},  # * matches anything
    )


def test_amr_similarity(requires_amr) -> None:
    """Test AMR semantic similarity."""
    from pytest_nlp import assert_amr_similarity, sentence_amr_similarity

    # Get Smatch F1 score
    score = sentence_amr_similarity(
        "The boy wants to go.",
        "The child wants to leave.",
    )
    assert score > 0

    # Assert sentences have similar AMR structure
    assert_amr_similarity(
        "The boy wants to go.",
        "The child wants to leave.",
        threshold=0.5,
    )
