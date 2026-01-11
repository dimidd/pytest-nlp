"""Tests for semantic similarity module."""

import pytest

from pytest_nlp import (
    assert_semantic_contains,
    assert_semantic_similarity,
    semantic_contains,
    semantic_similarity,
)


class TestSemanticSimilarity:
    """Tests for semantic_similarity function."""

    def test_identical_texts(self) -> None:
        """Identical texts should have high similarity."""
        score = semantic_similarity("hello world", "hello world")
        assert score > 0.99

    def test_similar_texts(self) -> None:
        """Semantically similar texts should have reasonable similarity."""
        score = semantic_similarity("The cat sat on the mat", "A cat is sitting on a rug")
        assert score > 0.5

    def test_dissimilar_texts(self) -> None:
        """Dissimilar texts should have low similarity."""
        score = semantic_similarity("The weather is nice today", "Quantum physics is complex")
        assert score < 0.5

    def test_different_metrics(self) -> None:
        """Test different similarity metrics."""
        text1 = "Hello world"
        text2 = "Hi there world"

        cosine = semantic_similarity(text1, text2, metric="cosine")
        dot = semantic_similarity(text1, text2, metric="dot")
        euclidean = semantic_similarity(text1, text2, metric="euclidean")

        # All should return valid scores
        assert 0 <= cosine <= 1
        assert euclidean >= 0  # Euclidean similarity is always positive
        # Dot product can be any value


class TestSemanticContains:
    """Tests for semantic_contains function."""

    def test_contains_high_similarity(self, drug_document: str) -> None:
        """Test finding a phrase with high semantic similarity."""
        is_contained, score = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            threshold=0.7,
        )
        assert is_contained
        assert score > 0.7

    def test_contains_semantic(self, drug_document: str) -> None:
        """Test finding semantically similar content."""
        is_contained, score = semantic_contains(
            query="Ibuprofen was stopped",
            document=drug_document,
            threshold=0.5,
            sentences=slice(-1, None),  # Last sentence only
        )
        assert is_contained, f"Query not semantically contained in document. Score: {score}"
        assert score > 0.5, f"Score is not greater than 0.5. Score: {score}"

    def test_not_contains(self, drug_document: str) -> None:
        """Test content that is not in document."""
        is_contained, score = semantic_contains(
            query="Patient underwent surgery",
            document=drug_document,
            threshold=0.7,
        )
        assert not is_contained
        assert score < 0.7

    def test_sentence_filter_last(self, drug_document: str) -> None:
        """Test filtering to last sentence only."""
        # Discontinuation is in last sentence
        is_contained, score = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=[-1],
            threshold=0.7,
        )
        assert is_contained

    def test_sentence_filter_first(self, drug_document: str) -> None:
        """Test filtering to first sentence only."""
        # Discontinuation is NOT in first sentence
        is_contained, _ = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=[0],
            threshold=0.8,
        )
        assert not is_contained

    def test_sentence_filter_slice(self, drug_document: str) -> None:
        """Test filtering with slice."""
        is_contained, _ = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=slice(-2, None),  # Last 2 sentences
            threshold=0.7,
        )
        assert is_contained


class TestAssertSemanticContains:
    """Tests for assert_semantic_contains function."""

    def test_assertion_passes(self, drug_document: str) -> None:
        """Test that assertion passes for matching content."""
        assert_semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            threshold=0.7,
        )

    def test_assertion_fails(self, drug_document: str) -> None:
        """Test that assertion fails for non-matching content."""
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_contains(
                query="Patient underwent surgery",
                document=drug_document,
                threshold=0.9,
            )
        assert "not semantically contained" in str(exc_info.value)

    def test_custom_message(self, drug_document: str) -> None:
        """Test custom error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_contains(
                query="Nonexistent content",
                document=drug_document,
                threshold=0.99,
                msg="Custom error message",
            )
        assert "Custom error message" in str(exc_info.value)


class TestAssertSemanticSimilarity:
    """Tests for assert_semantic_similarity function."""

    def test_assertion_passes(self) -> None:
        """Test that assertion passes for similar texts."""
        assert_semantic_similarity(
            text1="The cat sat on the mat",
            text2="A cat is sitting on a rug",
            threshold=0.5,
        )

    def test_assertion_fails(self) -> None:
        """Test that assertion fails for dissimilar texts."""
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_similarity(
                text1="Hello world",
                text2="Quantum mechanics",
                threshold=0.9,
            )
        assert "not semantically similar" in str(exc_info.value)

