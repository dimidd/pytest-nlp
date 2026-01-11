"""Model management with lazy loading and caching.

This module handles loading and caching of NLP models used by pytest-nlp:
- Sentence Transformers for semantic similarity
- spaCy for tokenization and dependency parsing
- Stanza for constituency parsing
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy
    import stanza
    from sentence_transformers import SentenceTransformer


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


def clear_model_cache() -> None:
    """Clear all cached models.

    Useful for testing or when memory needs to be freed.
    """
    get_embedding_model.cache_clear()
    get_spacy_model.cache_clear()
    get_stanza_pipeline.cache_clear()

