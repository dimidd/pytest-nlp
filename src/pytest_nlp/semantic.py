"""Semantic similarity module using sentence transformers.

This module provides functions for computing semantic similarity between
text segments and checking if a query is semantically contained in a document.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pytest_nlp.models import (
    ANY,
    SentenceMatch,
    _AnySentinel,
    get_embedding_model,
    get_spacy_model,
    split_sentences,
)


SimilarityMetric = Literal["cosine", "euclidean", "dot"]

# Type alias for sentence selection parameter
SentenceSelector = list[int] | slice | _AnySentinel | SentenceMatch | None


def _select_sentences(
    sentences: list[str],
    selector: list[int] | slice | SentenceMatch | None,
) -> list[str]:
    """Select a subset of sentences based on indices, slice, or pattern match.

    :param sentences: List of sentence strings.
    :param selector: Selector for sentences:
        - None: all sentences
        - list[int]: specific sentence indices
        - slice: slice of sentences
        - SentenceMatch: pattern-based selection
    :returns: Selected subset of sentences.
    """
    if selector is None:
        return sentences
    if isinstance(selector, slice):
        return sentences[selector]
    if isinstance(selector, SentenceMatch):
        return selector.select(sentences)
    return [sentences[i] for i in selector]


def _compute_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    metric: SimilarityMetric = "cosine",
) -> np.ndarray:
    """Compute similarity between query and document embeddings.

    :param query_embedding: Query embedding vector (1D).
    :param doc_embeddings: Document embeddings matrix (2D).
    :param metric: Similarity metric to use.
    :returns: Array of similarity scores.
    """
    if metric == "cosine":
        # Normalize vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9)
        return np.dot(doc_norms, query_norm)
    elif metric == "dot":
        return np.dot(doc_embeddings, query_embedding)
    elif metric == "euclidean":
        # Convert distance to similarity (higher is more similar)
        distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
        return 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def semantic_similarity(
    query: str,
    document: str,
    model_name: str = "all-MiniLM-L6-v2",
    metric: SimilarityMetric = "cosine",
    device: str | None = None,
    truncate_dim: int | None = None,
    prompt: str | None = None,
) -> float:
    """Compute semantic similarity between query and document.

    :param query: Query text.
    :param document: Document text.
    :param model_name: Name of the sentence-transformers model.
    :param metric: Similarity metric ('cosine', 'euclidean', 'dot').
    :param device: Device to run the model on.
    :param truncate_dim: Dimension to truncate embeddings to (for matryoshka models).
    :param prompt: Prompt prefix for Instructor-style embeddings.
    :returns: Similarity score between query and document.

    >>> score = semantic_similarity("hello world", "hi there world")
    >>> 0.0 <= score <= 1.0
    True
    """
    model = get_embedding_model(model_name, device=device, truncate_dim=truncate_dim)

    # Apply prompt if provided (for Instructor-style models)
    if prompt:
        query_text = f"{prompt} {query}"
        doc_text = f"{prompt} {document}"
    else:
        query_text = query
        doc_text = document

    embeddings = model.encode([query_text, doc_text])
    query_emb, doc_emb = embeddings[0], embeddings[1]

    similarities = _compute_similarity(query_emb, doc_emb.reshape(1, -1), metric)
    return float(similarities[0])


def semantic_contains(
    query: str,
    document: str,
    sentences: SentenceSelector = None,
    model_name: str = "all-MiniLM-L6-v2",
    metric: SimilarityMetric = "cosine",
    threshold: float = 0.7,
    device: str | None = None,
    truncate_dim: int | None = None,
    prompt: str | None = None,
    spacy_model: str = "en_core_web_sm",
) -> tuple[bool, float]:
    """Check if query is semantically contained in document.

    The document is split into sentences and the query is compared against
    each sentence. Returns True if any sentence exceeds the similarity threshold.

    :param query: Query text to search for.
    :param document: Document text to search in.
    :param sentences: Sentence selector - can be:
        - None: use all sentences (spaCy segmentation)
        - list[int]: specific sentence indices (e.g., [-2, -1] for last 2)
        - slice: slice of sentences
        - ANY: split on semicolons too, match if ANY sub-sentence matches
        - SentenceMatch: pattern-based selection (first/last/all matching)
    :param model_name: Name of the sentence-transformers model.
    :param metric: Similarity metric ('cosine', 'euclidean', 'dot').
    :param threshold: Minimum similarity score to consider a match.
    :param device: Device to run the model on.
    :param truncate_dim: Dimension to truncate embeddings to.
    :param prompt: Prompt prefix for Instructor-style embeddings.
    :param spacy_model: spaCy model for sentence segmentation.
    :returns: Tuple of (is_contained, max_similarity_score).

    >>> contains, score = semantic_contains("hello", "Hello there. Goodbye now.")
    >>> isinstance(contains, bool) and isinstance(score, float)
    True
    """
    # Segment document into sentences based on selector type
    if sentences is ANY:
        # Granular splitting with semicolons
        all_sentences = split_sentences(document, spacy_model=spacy_model, split_on_semicolon=True)
        selected_sentences = all_sentences
    else:
        # Standard spaCy sentence segmentation
        nlp = get_spacy_model(spacy_model)
        doc = nlp(document)
        all_sentences = [sent.text.strip() for sent in doc.sents]

        if not all_sentences:
            return False, 0.0

        # Select subset of sentences
        selected_sentences = _select_sentences(all_sentences, sentences)

    if not selected_sentences:
        return False, 0.0

    # Get embeddings
    model = get_embedding_model(model_name, device=device, truncate_dim=truncate_dim)

    if prompt:
        query_text = f"{prompt} {query}"
        sentence_texts = [f"{prompt} {s}" for s in selected_sentences]
    else:
        query_text = query
        sentence_texts = selected_sentences

    all_texts = [query_text] + sentence_texts
    embeddings = model.encode(all_texts)
    query_emb = embeddings[0]
    sentence_embs = embeddings[1:]

    # Compute similarities
    similarities = _compute_similarity(query_emb, sentence_embs, metric)
    max_similarity = float(np.max(similarities))

    return max_similarity >= threshold, max_similarity


def assert_semantic_contains(
    query: str,
    document: str,
    threshold: float = 0.7,
    sentences: SentenceSelector = None,
    model_name: str = "all-MiniLM-L6-v2",
    metric: SimilarityMetric = "cosine",
    device: str | None = None,
    truncate_dim: int | None = None,
    prompt: str | None = None,
    spacy_model: str = "en_core_web_sm",
    msg: str | None = None,
) -> None:
    """Assert that query is semantically contained in document.

    :param query: Query text to search for.
    :param document: Document text to search in.
    :param threshold: Minimum similarity score to consider a match.
    :param sentences: Sentence selector - can be:
        - None: use all sentences (spaCy segmentation)
        - list[int]: specific sentence indices
        - slice: slice of sentences
        - ANY: split on semicolons too, match if ANY sub-sentence matches
        - SentenceMatch: pattern-based selection (first/last/all matching)
    :param model_name: Name of the sentence-transformers model.
    :param metric: Similarity metric ('cosine', 'euclidean', 'dot').
    :param device: Device to run the model on.
    :param truncate_dim: Dimension to truncate embeddings to.
    :param prompt: Prompt prefix for Instructor-style embeddings.
    :param spacy_model: spaCy model for sentence segmentation.
    :param msg: Custom error message.
    :raises AssertionError: If query is not semantically contained in document.
    """
    is_contained, score = semantic_contains(
        query=query,
        document=document,
        sentences=sentences,
        model_name=model_name,
        metric=metric,
        threshold=threshold,
        device=device,
        truncate_dim=truncate_dim,
        prompt=prompt,
        spacy_model=spacy_model,
    )

    if not is_contained:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Query not semantically contained in document.\n"
            f"  Query: {query!r}\n"
            f"  Max similarity: {score:.4f} (threshold: {threshold})\n"
            f"  Sentences filter: {sentences}"
        )


def assert_semantic_similarity(
    text1: str,
    text2: str,
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
    metric: SimilarityMetric = "cosine",
    device: str | None = None,
    truncate_dim: int | None = None,
    prompt: str | None = None,
    msg: str | None = None,
) -> None:
    """Assert that two texts are semantically similar.

    :param text1: First text.
    :param text2: Second text.
    :param threshold: Minimum similarity score.
    :param model_name: Name of the sentence-transformers model.
    :param metric: Similarity metric ('cosine', 'euclidean', 'dot').
    :param device: Device to run the model on.
    :param truncate_dim: Dimension to truncate embeddings to.
    :param prompt: Prompt prefix for Instructor-style embeddings.
    :param msg: Custom error message.
    :raises AssertionError: If texts are not semantically similar.
    """
    score = semantic_similarity(
        query=text1,
        document=text2,
        model_name=model_name,
        metric=metric,
        device=device,
        truncate_dim=truncate_dim,
        prompt=prompt,
    )

    if score < threshold:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Texts not semantically similar.\n"
            f"  Text 1: {text1!r}\n"
            f"  Text 2: {text2!r}\n"
            f"  Similarity: {score:.4f} (threshold: {threshold})"
        )

