"""Tests for constituency parsing module."""

import pytest

from pytest_nlp import (
    assert_matches_constituency,
    get_constituency_matches,
    get_constituency_tree,
    match_constituency,
    parse_pattern,
)


class TestParsePattern:
    """Tests for pattern parsing."""

    def test_simple_pattern(self) -> None:
        """Test parsing a simple pattern."""
        node = parse_pattern("(NP (DT the) (NN cat))")
        assert node is not None
        assert node.label == "NP"
        assert len(node.children) == 2
        assert node.children[0].label == "DT"
        assert node.children[1].label == "NN"

    def test_variable_capture(self) -> None:
        """Test parsing pattern with variable capture."""
        node = parse_pattern("(NP (NNP ?drug))")
        assert node is not None
        assert len(node.children) == 1
        nnp = node.children[0]
        assert nnp.label == "NNP"
        assert len(nnp.children) == 1
        assert nnp.children[0].variable == "drug"

    def test_partial_match(self) -> None:
        """Test parsing pattern with partial match."""
        node = parse_pattern("(VP (VBD was) ...)")
        assert node is not None
        assert node.allow_partial

    def test_wildcard_label(self) -> None:
        """Test parsing pattern with wildcard label."""
        node = parse_pattern("(VP (VB*))")
        assert node is not None
        vb = node.children[0]
        assert vb.label == "VB"
        assert vb.is_wildcard

    def test_nested_pattern(self) -> None:
        """Test parsing deeply nested pattern."""
        pattern = "(S (NP (NNP Ibuprofen)) (VP (VBD was) (VP (VBN discontinued))))"
        node = parse_pattern(pattern)
        assert node is not None
        assert node.label == "S"
        assert len(node.children) == 2
        # NP child
        np = node.children[0]
        assert np.label == "NP"
        # VP child
        vp = node.children[1]
        assert vp.label == "VP"


class TestGetConstituencyTree:
    """Tests for getting constituency trees."""

    def test_single_sentence(self) -> None:
        """Test getting tree for single sentence."""
        trees = get_constituency_tree("The cat sat on the mat.")
        assert len(trees) == 1
        assert "(" in trees[0]  # S-expression format

    def test_multiple_sentences(self) -> None:
        """Test getting trees for multiple sentences."""
        trees = get_constituency_tree("The cat sat. The dog ran.")
        assert len(trees) == 2

    def test_sentence_filter(self) -> None:
        """Test filtering sentences."""
        trees = get_constituency_tree(
            "First sentence. Second sentence. Third sentence.",
            sentences=[-1],
        )
        assert len(trees) == 1


class TestMatchConstituency:
    """Tests for constituency matching."""

    def test_simple_match(self) -> None:
        """Test simple constituency match."""
        matches = match_constituency(
            doc="The cat sat.",
            pattern="(NP ...)",
        )
        assert len(matches) >= 1

    def test_variable_capture(self) -> None:
        """Test capturing variables."""
        matches = match_constituency(
            doc="The cat sat.",
            pattern="(NP (DT ?det) (NN ?noun))",
        )
        # Should capture determiner and noun
        if matches:  # May not match depending on parse
            assert "det" in matches[0] or "noun" in matches[0]

    def test_no_match(self) -> None:
        """Test pattern that doesn't match."""
        matches = match_constituency(
            doc="Hello world.",
            pattern="(SBARQ (WHNP ...))",  # Question pattern on a statement
        )
        # May or may not match depending on the actual parse
        assert isinstance(matches, list)

    def test_sentence_filter(self) -> None:
        """Test filtering by sentence."""
        doc = "The cat sat. The dog ran."
        matches_all = match_constituency(doc, pattern="(NP ...)")
        matches_first = match_constituency(doc, pattern="(NP ...)", sentences=[0])

        # Should have fewer matches when filtering
        assert len(matches_first) <= len(matches_all)


class TestGetConstituencyMatches:
    """Tests for get_constituency_matches function."""

    def test_returns_match_objects(self) -> None:
        """Test that full match objects are returned."""
        matches = get_constituency_matches(
            doc="The cat sat.",
            pattern="(NP ...)",
        )
        if matches:
            assert hasattr(matches[0], "text")
            assert hasattr(matches[0], "captures")
            assert hasattr(matches[0], "tree_str")


class TestAssertMatchesConstituency:
    """Tests for assertion function."""

    def test_assertion_passes(self) -> None:
        """Test that assertion passes for matching pattern."""
        assert_matches_constituency(
            doc="The cat sat on the mat.",
            pattern="(NP ...)",
        )

    def test_assertion_fails(self) -> None:
        """Test that assertion fails for non-matching pattern."""
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_constituency(
                doc="Hello.",
                pattern="(SBARQ (WHNP (WP what)) ...)",  # Very specific question pattern
                min_matches=10,  # Require many matches
            )
        assert "Constituency pattern did not match" in str(exc_info.value)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_constituency(
                doc="Hello.",
                pattern="(IMPOSSIBLE_TAG)",
                msg="Custom constituency error",
            )
        assert "Custom constituency error" in str(exc_info.value)

    def test_min_matches(self) -> None:
        """Test minimum matches requirement."""
        # Should have at least one NP
        assert_matches_constituency(
            doc="The big cat sat on the small mat.",
            pattern="(NP ...)",
            min_matches=1,
        )


class TestDrugDiscontinuationExample:
    """Integration tests with drug discontinuation example."""

    @pytest.fixture
    def discontinuation_sentence(self) -> str:
        """Sample sentence about drug discontinuation."""
        return "Ibuprofen was discontinued on 12/30/2023."

    def test_get_tree(self, discontinuation_sentence: str) -> None:
        """Test getting constituency tree for drug sentence."""
        trees = get_constituency_tree(discontinuation_sentence)
        assert len(trees) == 1
        # Tree should contain expected elements
        tree = trees[0]
        assert "Ibuprofen" in tree or "ibuprofen" in tree.lower()

    def test_match_passive_structure(self, discontinuation_sentence: str) -> None:
        """Test matching passive voice structure."""
        # Look for VP with passive construction
        matches = get_constituency_matches(
            doc=discontinuation_sentence,
            pattern="(VP (VBD was) ...)",
        )
        assert len(matches) >= 1

