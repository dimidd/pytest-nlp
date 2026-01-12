"""SpaCy matcher wrappers for token, phrase, and dependency pattern matching.

This module provides functions for pattern matching using spaCy's
Matcher, PhraseMatcher, and DependencyMatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spacy
from spacy.matcher import DependencyMatcher, Matcher, PhraseMatcher
from spacy.tokens import Doc, Span

from pytest_nlp.models import ANY, _AnySentinel, get_spacy_model, split_sentences


# Type alias for sentence selection parameter
SentenceSelector = list[int] | slice | _AnySentinel | None


@dataclass
class MatchResult:
    """Result of a pattern match.

    :param text: Matched text span.
    :param start: Start character offset.
    :param end: End character offset.
    :param start_token: Start token index.
    :param end_token: End token index.
    :param label: Pattern label (if any).
    """

    text: str
    start: int
    end: int
    start_token: int
    end_token: int
    label: str | None = None

    @classmethod
    def from_span(cls, span: Span, label: str | None = None) -> "MatchResult":
        """Create MatchResult from a spaCy Span.

        :param span: spaCy Span object.
        :param label: Optional label for the match.
        :returns: MatchResult instance.
        """
        return cls(
            text=span.text,
            start=span.start_char,
            end=span.end_char,
            start_token=span.start,
            end_token=span.end,
            label=label,
        )


def _get_filtered_doc(
    text: str,
    sentences: SentenceSelector,
    nlp: spacy.Language,
) -> tuple[Doc, str]:
    """Process text and filter to selected sentences.

    :param text: Input text.
    :param sentences: Sentence selector - indices, slice, ANY, or None.
    :param nlp: spaCy Language model.
    :returns: Tuple of (filtered Doc, filtered text).
    """
    doc = nlp(text)

    if sentences is None:
        return doc, text

    all_sents = list(doc.sents)

    if isinstance(sentences, slice):
        selected_sents = all_sents[sentences]
    elif isinstance(sentences, list):
        selected_sents = [all_sents[i] for i in sentences]
    else:
        # ANY or unknown - return full doc (ANY handling done at higher level)
        return doc, text

    if not selected_sents:
        # Return empty doc
        return nlp(""), ""

    # Combine selected sentences
    filtered_text = " ".join(sent.text for sent in selected_sents)
    filtered_doc = nlp(filtered_text)
    return filtered_doc, filtered_text


def _get_granular_sentences(
    text: str,
    spacy_model: str = "en_core_web_sm",
) -> list[str]:
    """Get granular sentences split on semicolons.

    :param text: Input text.
    :param spacy_model: spaCy model name for sentence segmentation.
    :returns: List of sentence strings.
    """
    return split_sentences(text, spacy_model=spacy_model, split_on_semicolon=True)


def match_tokens(
    doc: str,
    patterns: list[list[dict[str, Any]]],
    sentences: SentenceSelector = None,
    spacy_model: str = "en_core_web_sm",
) -> list[MatchResult]:
    """Match token patterns in a document using spaCy Matcher.

    :param doc: Document text to search in.
    :param patterns: List of token patterns (each pattern is a list of token specs).
    :param sentences: Sentence selector - can be:
        - None: use all sentences
        - list[int]: specific sentence indices
        - slice: slice of sentences
        - ANY: split on semicolons too, match across all sub-sentences
    :param spacy_model: spaCy model to use.
    :returns: List of MatchResult objects for each match.

    >>> results = match_tokens("The quick brown fox", [[{"LOWER": "quick"}]])
    >>> len(results) > 0
    True
    """
    nlp = get_spacy_model(spacy_model)

    # Handle ANY sentinel - process each granular sentence
    if sentences is ANY:
        granular_sentences = _get_granular_sentences(doc, spacy_model)
        all_results: list[MatchResult] = []
        for sentence in granular_sentences:
            all_results.extend(match_tokens(sentence, patterns, sentences=None, spacy_model=spacy_model))
        return all_results

    spacy_doc, _ = _get_filtered_doc(doc, sentences, nlp)

    matcher = Matcher(nlp.vocab)

    for i, pattern in enumerate(patterns):
        matcher.add(f"PATTERN_{i}", [pattern])

    matches = matcher(spacy_doc)
    results: list[MatchResult] = []

    for match_id, start, end in matches:
        span = spacy_doc[start:end]
        label = nlp.vocab.strings[match_id]
        results.append(MatchResult.from_span(span, label))

    return results


def match_phrases(
    doc: str,
    phrases: list[str],
    sentences: SentenceSelector = None,
    spacy_model: str = "en_core_web_sm",
    attr: str = "ORTH",
) -> list[MatchResult]:
    """Match phrase patterns in a document using spaCy PhraseMatcher.

    :param doc: Document text to search in.
    :param phrases: List of phrases to match.
    :param sentences: Sentence selector - can be:
        - None: use all sentences
        - list[int]: specific sentence indices
        - slice: slice of sentences
        - ANY: split on semicolons too, match across all sub-sentences
    :param spacy_model: spaCy model to use.
    :param attr: Token attribute to match on (e.g., 'ORTH', 'LOWER', 'LEMMA').
    :returns: List of MatchResult objects for each match.

    >>> results = match_phrases("The quick brown fox", ["quick brown"])
    >>> len(results) > 0
    True
    """
    nlp = get_spacy_model(spacy_model)

    # Handle ANY sentinel - process each granular sentence
    if sentences is ANY:
        granular_sentences = _get_granular_sentences(doc, spacy_model)
        all_results: list[MatchResult] = []
        for sentence in granular_sentences:
            all_results.extend(match_phrases(sentence, phrases, sentences=None, spacy_model=spacy_model, attr=attr))
        return all_results

    spacy_doc, _ = _get_filtered_doc(doc, sentences, nlp)

    matcher = PhraseMatcher(nlp.vocab, attr=attr)
    phrase_patterns = list(nlp.pipe(phrases))
    matcher.add("PHRASES", phrase_patterns)

    matches = matcher(spacy_doc)
    results: list[MatchResult] = []

    for _, start, end in matches:
        span = spacy_doc[start:end]
        results.append(MatchResult.from_span(span, "PHRASES"))

    return results


def match_dependency(
    doc: str,
    patterns: list[dict[str, Any]],
    sentences: SentenceSelector = None,
    spacy_model: str = "en_core_web_sm",
) -> list[MatchResult]:
    """Match dependency patterns in a document using spaCy DependencyMatcher.

    :param doc: Document text to search in.
    :param patterns: Dependency pattern (list of node specifications).
    :param sentences: Sentence selector - can be:
        - None: use all sentences
        - list[int]: specific sentence indices
        - slice: slice of sentences
        - ANY: split on semicolons too, match across all sub-sentences
    :param spacy_model: spaCy model to use.
    :returns: List of MatchResult objects for each match.

    >>> patterns = [
    ...     {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    ...     {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
    ...      "RIGHT_ATTRS": {"DEP": "nsubj"}}
    ... ]
    >>> results = match_dependency("The cat sat on the mat", patterns)
    >>> isinstance(results, list)
    True
    """
    nlp = get_spacy_model(spacy_model)

    # Handle ANY sentinel - process each granular sentence
    if sentences is ANY:
        granular_sentences = _get_granular_sentences(doc, spacy_model)
        all_results: list[MatchResult] = []
        for sentence in granular_sentences:
            all_results.extend(match_dependency(sentence, patterns, sentences=None, spacy_model=spacy_model))
        return all_results

    spacy_doc, _ = _get_filtered_doc(doc, sentences, nlp)

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("DEPENDENCY_PATTERN", [patterns])

    matches = matcher(spacy_doc)
    results: list[MatchResult] = []

    for _, token_ids in matches:
        if token_ids:
            # Create a span from the matched tokens
            start = min(token_ids)
            end = max(token_ids) + 1
            span = spacy_doc[start:end]
            results.append(MatchResult.from_span(span, "DEPENDENCY_PATTERN"))

    return results


def assert_matches_tokens(
    doc: str,
    patterns: list[list[dict[str, Any]]],
    sentences: SentenceSelector = None,
    spacy_model: str = "en_core_web_sm",
    min_matches: int = 1,
    msg: str | None = None,
) -> None:
    """Assert that token patterns match in the document.

    :param doc: Document text to search in.
    :param patterns: List of token patterns.
    :param sentences: Sentence selector - indices, slice, ANY, or None.
    :param spacy_model: spaCy model to use.
    :param min_matches: Minimum number of matches required.
    :param msg: Custom error message.
    :raises AssertionError: If patterns do not match.
    """
    results = match_tokens(doc, patterns, sentences, spacy_model)

    if len(results) < min_matches:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Token patterns did not match.\n"
            f"  Document: {doc!r}\n"
            f"  Patterns: {patterns}\n"
            f"  Expected at least {min_matches} match(es), found {len(results)}\n"
            f"  Sentences filter: {sentences}"
        )


def assert_matches_phrases(
    doc: str,
    phrases: list[str],
    sentences: SentenceSelector = None,
    spacy_model: str = "en_core_web_sm",
    attr: str = "ORTH",
    min_matches: int = 1,
    msg: str | None = None,
) -> None:
    """Assert that phrases match in the document.

    :param doc: Document text to search in.
    :param phrases: List of phrases to match.
    :param sentences: Sentence selector - indices, slice, ANY, or None.
    :param spacy_model: spaCy model to use.
    :param attr: Token attribute to match on.
    :param min_matches: Minimum number of matches required.
    :param msg: Custom error message.
    :raises AssertionError: If phrases do not match.
    """
    results = match_phrases(doc, phrases, sentences, spacy_model, attr)

    if len(results) < min_matches:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Phrase patterns did not match.\n"
            f"  Document: {doc!r}\n"
            f"  Phrases: {phrases}\n"
            f"  Expected at least {min_matches} match(es), found {len(results)}\n"
            f"  Sentences filter: {sentences}"
        )


def assert_matches_dependency(
    doc: str,
    patterns: list[dict[str, Any]],
    sentences: SentenceSelector = None,
    spacy_model: str = "en_core_web_sm",
    min_matches: int = 1,
    msg: str | None = None,
) -> None:
    """Assert that dependency patterns match in the document.

    :param doc: Document text to search in.
    :param patterns: Dependency pattern (list of node specifications).
    :param sentences: Sentence selector - indices, slice, ANY, or None.
    :param spacy_model: spaCy model to use.
    :param min_matches: Minimum number of matches required.
    :param msg: Custom error message.
    :raises AssertionError: If dependency patterns do not match.
    """
    results = match_dependency(doc, patterns, sentences, spacy_model)

    if len(results) < min_matches:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Dependency patterns did not match.\n"
            f"  Document: {doc!r}\n"
            f"  Patterns: {patterns}\n"
            f"  Expected at least {min_matches} match(es), found {len(results)}\n"
            f"  Sentences filter: {sentences}"
        )

