"""Model management with lazy loading and caching.

This module handles loading and caching of NLP models used by pytest-nlp:
- Sentence Transformers for semantic similarity
- spaCy for tokenization and dependency parsing
- Stanza for constituency parsing
- AMR parsing via amrlib
- Sentence splitting utilities
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Final


# =============================================================================
# Model Configuration Defaults
# =============================================================================

DEFAULT_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
"""Default sentence transformer model for semantic similarity."""

DEFAULT_SPACY_MODEL: Final[str] = "en_core_web_sm"
"""Default spaCy model for tokenization and dependency parsing."""

DEFAULT_STANZA_LANG: Final[str] = "en"
"""Default language for Stanza pipeline."""

DEFAULT_STANZA_PROCESSORS: Final[str] = "tokenize,pos,constituency"
"""Default processors for Stanza pipeline."""

DEFAULT_AMR_MODEL: Final[str] = "parse_xfm_bart_base"
"""Default AMR parser model (stog - sentence to graph).

Available models from https://github.com/bjascob/amrlib-models:
- ``parse_xfm_bart_large``: Best quality (83.7 SMATCH, 1.4GB, 17/sec)
- ``parse_xfm_bart_base``: Good balance (82.3 SMATCH, 492MB, 31/sec) [DEFAULT]
- ``parse_spring``: Alternative architecture (83.5 SMATCH, 1.5GB, 14/sec)
- ``parse_t5``: T5-based (81.9 SMATCH, 785MB, 11/sec)
"""

if TYPE_CHECKING:
    import spacy
    import stanza
    from sentence_transformers import SentenceTransformer


class _AnySentinel:
    """Sentinel class for the ANY sentence matcher.

    When passed as the `sentences` parameter, indicates that the text should
    be split into granular sentences (including on semicolons) and the match
    should pass if ANY sentence matches the query.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return string representation."""
        return "ANY"


ANY: Final[_AnySentinel] = _AnySentinel()
"""Sentinel indicating granular sentence splitting with any-match semantics.

When used as the `sentences` parameter:
1. Text is split into fine-grained sentences (on periods, semicolons, etc.)
2. The assertion passes if ANY sentence matches the query

Example::

    >>> from pytest_nlp import assert_semantic_contains, ANY
    >>> text = "Patient prescribed 20mg; later reduced to 10mg."
    >>> assert_semantic_contains("dosage reduced", text, sentences=ANY)
"""


# Pattern for splitting sentences on semicolons while preserving whitespace handling
_SEMICOLON_SPLIT_PATTERN = re.compile(r";\s*")

# Pattern for splitting on comma + coordinating conjunction (joins independent clauses)
# Matches: ", and ", ", but ", ", or ", ", so ", ", yet ", ", nor "
_COMMA_CONJUNCTION_PATTERN = re.compile(r",\s*(?:and|but|or|so|yet|nor)\s+", re.IGNORECASE)


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
    truncate_dim: int | None = None,
) -> "SentenceTransformer":
    """Load and cache a sentence transformer embedding model.

    :param model_name: Name of the sentence-transformers model to load.
    :param device: Device to load the model on (e.g., 'cpu', 'cuda').
        If None, uses the default device.
    :param truncate_dim: Dimension to truncate embeddings to (for matryoshka models).
    :returns: The loaded SentenceTransformer model.

    >>> model = get_embedding_model("all-MiniLM-L6-v2")
    >>> model.get_sentence_embedding_dimension() > 0
    True
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, truncate_dim=truncate_dim)
    return model


@lru_cache(maxsize=4)
def get_spacy_model(model_name: str = "en_core_web_sm") -> "spacy.Language":
    """Load and cache a spaCy language model.

    :param model_name: Name of the spaCy model to load.
    :returns: The loaded spaCy Language model.

    >>> nlp = get_spacy_model("en_core_web_sm")
    >>> doc = nlp("Hello world")
    >>> len(doc) > 0
    True
    """
    import spacy

    return spacy.load(model_name)


@lru_cache(maxsize=4)
def get_stanza_pipeline(
    lang: str = "en",
    processors: str = "tokenize,pos,constituency",
) -> "stanza.Pipeline":
    """Load and cache a Stanza NLP pipeline.

    :param lang: Language code for the Stanza model.
    :param processors: Comma-separated list of processors to load.
    :returns: The loaded Stanza Pipeline.

    >>> pipeline = get_stanza_pipeline("en")
    >>> doc = pipeline("Hello world")
    >>> len(doc.sentences) > 0
    True
    """
    import stanza

    # Download model if not present (silent if already downloaded)
    stanza.download(lang, processors=processors, verbose=False)
    return stanza.Pipeline(lang=lang, processors=processors, verbose=False)


def _split_on_patterns(
    sentences: list[str],
    split_semicolon: bool,
    split_comma_conjunction: bool,
) -> list[str]:
    """Apply splitting patterns to a list of sentences.

    :param sentences: List of sentence strings.
    :param split_semicolon: Whether to split on semicolons.
    :param split_comma_conjunction: Whether to split on comma + conjunction.
    :returns: List of split sentence strings.
    """
    result: list[str] = []

    for sentence in sentences:
        parts = [sentence]

        # Split on semicolons first
        if split_semicolon and ";" in sentence:
            new_parts: list[str] = []
            for part in parts:
                new_parts.extend(_SEMICOLON_SPLIT_PATTERN.split(part))
            parts = new_parts

        # Then split on comma + conjunction
        if split_comma_conjunction:
            new_parts = []
            for part in parts:
                if _COMMA_CONJUNCTION_PATTERN.search(part):
                    new_parts.extend(_COMMA_CONJUNCTION_PATTERN.split(part))
                else:
                    new_parts.append(part)
            parts = new_parts

        # Filter empty parts and strip whitespace
        result.extend(p.strip() for p in parts if p.strip())

    return result


def split_sentences(
    text: str,
    spacy_model: str = "en_core_web_sm",
    split_on_semicolon: bool = True,
    split_on_comma_conjunction: bool = True,
) -> list[str]:
    """Split text into sentences with granular splitting options.

    Uses spaCy for initial sentence segmentation, then optionally splits
    further on semicolons and comma + coordinating conjunctions for more
    granular sentence boundaries.

    :param text: Text to split into sentences.
    :param spacy_model: spaCy model to use for sentence segmentation.
    :param split_on_semicolon: Whether to additionally split on semicolons.
    :param split_on_comma_conjunction: Whether to split on comma followed by
        coordinating conjunctions (and, but, or, so, yet, nor).
    :returns: List of sentence strings, stripped of leading/trailing whitespace.

    >>> sentences = split_sentences("Hello world. Goodbye.")
    >>> len(sentences)
    2

    >>> text = "Prescribed 20mg daily; later reduced to 10mg."
    >>> sentences = split_sentences(text)
    >>> len(sentences)
    2
    >>> sentences[0]
    'Prescribed 20mg daily'
    >>> sentences[1]
    'later reduced to 10mg.'

    >>> text = "Prescribed 20mg daily, and later a taper was initiated."
    >>> sentences = split_sentences(text)
    >>> len(sentences)
    2
    >>> sentences[0]
    'Prescribed 20mg daily'
    >>> sentences[1]
    'later a taper was initiated.'
    """
    nlp = get_spacy_model(spacy_model)
    doc = nlp(text)

    # Get spaCy sentence boundaries
    spacy_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not split_on_semicolon and not split_on_comma_conjunction:
        return spacy_sentences

    return _split_on_patterns(spacy_sentences, split_on_semicolon, split_on_comma_conjunction)


def clear_model_cache() -> None:
    """Clear all cached models.

    Useful for testing or when memory needs to be freed.
    """
    get_embedding_model.cache_clear()
    get_spacy_model.cache_clear()
    get_stanza_pipeline.cache_clear()

