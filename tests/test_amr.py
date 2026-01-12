"""Tests for AMR (Abstract Meaning Representation) module."""

import pytest

# Skip all tests if amrlib is not installed
pytest.importorskip("amrlib")

from pytest_nlp.amr import (
    AMRGraph,
    AMRMatch,
    assert_amr_pattern,
    assert_amr_similarity,
    assert_has_concept,
    assert_has_role,
    assert_is_negated,
    assert_not_negated,
    find_concepts,
    match_amr_pattern,
    parse_amr,
    sentence_amr_similarity,
)


@pytest.fixture(scope="module")
def simple_graph() -> AMRGraph:
    """Parse a simple sentence for testing."""
    return parse_amr("The boy wants to go.")


@pytest.fixture(scope="module")
def negated_graph() -> AMRGraph:
    """Parse a negated sentence for testing."""
    return parse_amr("The boy does not want to go.")


class TestParseAMR:
    """Tests for parse_amr function."""

    def test_parse_simple_sentence(self) -> None:
        """Test parsing a simple sentence."""
        graph = parse_amr("The dog barked.")
        assert isinstance(graph, AMRGraph)
        assert graph.sentence == "The dog barked."

    def test_parse_returns_concepts(self, simple_graph: AMRGraph) -> None:
        """Test that parsed graph contains concepts."""
        concepts = [c for _, c in simple_graph.concepts]
        assert len(concepts) >= 2  # At least want-01 and go-02

    def test_parse_returns_roles(self, simple_graph: AMRGraph) -> None:
        """Test that parsed graph contains roles."""
        assert len(simple_graph.roles) >= 1


class TestAMRGraph:
    """Tests for AMRGraph wrapper class."""

    def test_has_concept(self, simple_graph: AMRGraph) -> None:
        """Test has_concept method."""
        assert simple_graph.has_concept("want-01")
        assert not simple_graph.has_concept("nonexistent-99")

    def test_has_role(self, simple_graph: AMRGraph) -> None:
        """Test has_role method."""
        # want-01 should have ARG0 (wanter) and ARG1 (thing wanted)
        assert simple_graph.has_role(":ARG0")
        assert simple_graph.has_role(":ARG1")

    def test_has_role_without_colon(self, simple_graph: AMRGraph) -> None:
        """Test has_role with role name without leading colon."""
        assert simple_graph.has_role("ARG0")

    def test_is_negated_false(self, simple_graph: AMRGraph) -> None:
        """Test is_negated returns False for non-negated sentence."""
        assert not simple_graph.is_negated()

    def test_is_negated_true(self, negated_graph: AMRGraph) -> None:
        """Test is_negated returns True for negated sentence."""
        assert negated_graph.is_negated()

    def test_get_role_fillers(self, simple_graph: AMRGraph) -> None:
        """Test get_role_fillers method."""
        fillers = simple_graph.get_role_fillers(":ARG1", "want-01")
        assert len(fillers) >= 1

    def test_str_representation(self, simple_graph: AMRGraph) -> None:
        """Test string representation of graph."""
        amr_str = str(simple_graph)
        assert "want-01" in amr_str
        assert "(" in amr_str  # AMR format uses parentheses


class TestFindConcepts:
    """Tests for find_concepts function."""

    def test_find_exact_concept(self, simple_graph: AMRGraph) -> None:
        """Test finding an exact concept."""
        matches = find_concepts(simple_graph, "want-01")
        assert len(matches) >= 1
        assert matches[0].concept == "want-01"

    def test_find_wildcard_concept(self, simple_graph: AMRGraph) -> None:
        """Test finding concepts with wildcard."""
        matches = find_concepts(simple_graph, "want-*")
        assert len(matches) >= 1

    def test_find_nonexistent_concept(self, simple_graph: AMRGraph) -> None:
        """Test finding a concept that doesn't exist."""
        matches = find_concepts(simple_graph, "nonexistent-99")
        assert len(matches) == 0

    def test_match_returns_roles(self, simple_graph: AMRGraph) -> None:
        """Test that matches include role information."""
        matches = find_concepts(simple_graph, "want-01")
        assert len(matches) >= 1
        assert isinstance(matches[0].roles, dict)


class TestMatchAMRPattern:
    """Tests for match_amr_pattern function."""

    def test_match_concept_only(self, simple_graph: AMRGraph) -> None:
        """Test matching concept without roles."""
        matches = match_amr_pattern(simple_graph, "want-01")
        assert len(matches) >= 1

    def test_match_with_role(self, simple_graph: AMRGraph) -> None:
        """Test matching concept with specific role."""
        matches = match_amr_pattern(simple_graph, "want-01", {":ARG0": "*"})
        assert len(matches) >= 1

    def test_match_with_role_value(self) -> None:
        """Test matching concept with role value."""
        graph = parse_amr("The boy wants the girl to believe him.")
        matches = match_amr_pattern(graph, "want-01", {":ARG0": "boy"})
        assert len(matches) >= 1

    def test_no_match_wrong_role(self, simple_graph: AMRGraph) -> None:
        """Test no match when role doesn't exist."""
        matches = match_amr_pattern(simple_graph, "want-01", {":ARG99": "*"})
        assert len(matches) == 0


class TestAssertHasConcept:
    """Tests for assert_has_concept function."""

    def test_assertion_passes(self) -> None:
        """Test assertion passes for existing concept."""
        assert_has_concept("The boy wants to go.", "want-01")

    def test_assertion_fails(self) -> None:
        """Test assertion fails for missing concept."""
        with pytest.raises(AssertionError) as exc_info:
            assert_has_concept("The dog barked.", "want-01")
        assert "not found" in str(exc_info.value)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_has_concept("Hello.", "want-01", msg="Custom error")
        assert "Custom error" in str(exc_info.value)


class TestAssertHasRole:
    """Tests for assert_has_role function."""

    def test_assertion_passes(self) -> None:
        """Test assertion passes for existing role."""
        assert_has_role("The boy wants to go.", ":ARG0")

    def test_assertion_fails(self) -> None:
        """Test assertion fails for missing role."""
        with pytest.raises(AssertionError) as exc_info:
            assert_has_role("Hello.", ":ARG99")
        assert "not found" in str(exc_info.value)

    def test_with_source_concept(self) -> None:
        """Test assertion with source concept filter."""
        assert_has_role("The boy wants to go.", ":ARG0", source_concept="want-01")


class TestAssertNegation:
    """Tests for negation assertion functions."""

    def test_assert_is_negated_passes(self) -> None:
        """Test assert_is_negated passes for negated sentence."""
        assert_is_negated("The boy does not want to go.")

    def test_assert_is_negated_fails(self) -> None:
        """Test assert_is_negated fails for non-negated sentence."""
        with pytest.raises(AssertionError) as exc_info:
            assert_is_negated("The boy wants to go.")
        assert "not negated" in str(exc_info.value)

    def test_assert_not_negated_passes(self) -> None:
        """Test assert_not_negated passes for non-negated sentence."""
        assert_not_negated("The boy wants to go.")

    def test_assert_not_negated_fails(self) -> None:
        """Test assert_not_negated fails for negated sentence."""
        with pytest.raises(AssertionError) as exc_info:
            assert_not_negated("The boy does not want to go.")
        assert "is negated" in str(exc_info.value)


class TestAssertAMRPattern:
    """Tests for assert_amr_pattern function."""

    def test_assertion_passes(self) -> None:
        """Test assertion passes for matching pattern."""
        assert_amr_pattern(
            "The boy wants to go.",
            concept="want-01",
            roles={":ARG0": "*"},
        )

    def test_assertion_fails(self) -> None:
        """Test assertion fails for non-matching pattern."""
        with pytest.raises(AssertionError) as exc_info:
            assert_amr_pattern(
                "Hello.",
                concept="want-01",
            )
        assert "not matched" in str(exc_info.value)

    def test_min_matches(self) -> None:
        """Test min_matches parameter."""
        with pytest.raises(AssertionError):
            assert_amr_pattern(
                "The boy wants to go.",
                concept="want-01",
                min_matches=10,
            )


class TestAMRSimilarity:
    """Tests for AMR similarity functions."""

    def test_identical_sentences(self) -> None:
        """Test similarity of identical sentences."""
        score = sentence_amr_similarity(
            "The boy wants to go.",
            "The boy wants to go.",
        )
        assert score > 0.9

    def test_similar_sentences(self) -> None:
        """Test similarity of semantically similar sentences."""
        score = sentence_amr_similarity(
            "The boy wants to go.",
            "The child wants to leave.",
        )
        assert score > 0.3  # Should have some structural similarity

    def test_dissimilar_sentences(self) -> None:
        """Test similarity of dissimilar sentences."""
        score = sentence_amr_similarity(
            "The boy wants to go.",
            "It is raining heavily today.",
        )
        assert score < 0.5


class TestAssertAMRSimilarity:
    """Tests for assert_amr_similarity function."""

    def test_assertion_passes(self) -> None:
        """Test assertion passes for similar sentences."""
        assert_amr_similarity(
            "The boy wants to go.",
            "The child wants to leave.",
            threshold=0.3,
        )

    def test_assertion_fails(self) -> None:
        """Test assertion fails for dissimilar sentences."""
        with pytest.raises(AssertionError) as exc_info:
            assert_amr_similarity(
                "Hello world.",
                "It is raining heavily today.",
                threshold=0.9,
            )
        assert "not similar enough" in str(exc_info.value)


class TestDrugDiscontinuationScenario:
    """Integration tests with drug discontinuation example."""

    def test_discontinue_concept(self) -> None:
        """Test finding discontinue concept."""
        sentence = "The doctor discontinued the medication."
        assert_has_concept(sentence, "discontinue-01")

    def test_discontinue_roles(self) -> None:
        """Test discontinue has correct roles."""
        sentence = "The doctor discontinued the medication."
        # Should have ARG0 (agent) and ARG1 (thing discontinued)
        assert_has_role(sentence, ":ARG0", source_concept="discontinue-01")
        assert_has_role(sentence, ":ARG1", source_concept="discontinue-01")

    def test_negated_discontinue(self) -> None:
        """Test detecting negated discontinuation."""
        sentence = "The doctor did not discontinue the medication."
        assert_is_negated(sentence, "discontinue-01")

    def test_prescribe_concept(self) -> None:
        """Test finding prescribe concept."""
        sentence = "The doctor prescribed ibuprofen."
        # Note: AMR might use different framesets
        graph = parse_amr(sentence)
        # Should have some concept related to prescribing
        concepts = [c for _, c in graph.concepts]
        assert any("prescribe" in c or "doctor" in c for c in concepts)

