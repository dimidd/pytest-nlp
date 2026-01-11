"""Pytest plugin hooks and configuration for pytest-nlp.

This module provides pytest hooks for loading configuration and
managing NLP model lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


@dataclass
class NLPConfig:
    """Configuration for pytest-nlp.

    :param embedding_model: Name of the sentence-transformers model.
    :param spacy_model: Name of the spaCy model.
    :param stanza_lang: Language code for Stanza.
    :param similarity_threshold: Default similarity threshold.
    :param similarity_metric: Default similarity metric.
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    stanza_lang: str = "en"
    similarity_threshold: float = 0.7
    similarity_metric: str = "cosine"


# Global config instance
_config: NLPConfig | None = None


def get_config() -> NLPConfig:
    """Get the current NLP configuration.

    :returns: NLPConfig instance with current settings.
    """
    global _config
    if _config is None:
        _config = NLPConfig()
    return _config


def pytest_addoption(parser: "Parser") -> None:
    """Add pytest-nlp configuration options."""
    parser.addini(
        "nlp_embedding_model",
        "Name of the sentence-transformers embedding model",
        default="all-MiniLM-L6-v2",
    )
    parser.addini(
        "nlp_spacy_model",
        "Name of the spaCy language model",
        default="en_core_web_sm",
    )
    parser.addini(
        "nlp_stanza_lang",
        "Language code for Stanza",
        default="en",
    )
    parser.addini(
        "nlp_similarity_threshold",
        "Default similarity threshold for semantic matching",
        default="0.7",
    )
    parser.addini(
        "nlp_similarity_metric",
        "Default similarity metric (cosine, euclidean, dot)",
        default="cosine",
    )


def pytest_configure(config: "Config") -> None:
    """Configure pytest-nlp based on pytest config."""
    global _config

    _config = NLPConfig(
        embedding_model=config.getini("nlp_embedding_model"),
        spacy_model=config.getini("nlp_spacy_model"),
        stanza_lang=config.getini("nlp_stanza_lang"),
        similarity_threshold=float(config.getini("nlp_similarity_threshold")),
        similarity_metric=config.getini("nlp_similarity_metric"),
    )


def pytest_unconfigure(config: "Config") -> None:
    """Clean up when pytest finishes."""
    from pytest_nlp.models import clear_model_cache

    clear_model_cache()


@pytest.fixture(scope="session")
def nlp_config() -> NLPConfig:
    """Pytest fixture providing the NLP configuration.

    :returns: NLPConfig instance with current settings.
    """
    return get_config()

