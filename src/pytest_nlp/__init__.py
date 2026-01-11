"""pytest-nlp: A pytest plugin for evaluating free-text outputs using NLP.

This plugin provides assertions for:
- Semantic similarity using sentence transformers
- Token, phrase, and dependency matching using spaCy
- Constituency parsing patterns using Stanza
"""

from pytest_nlp.constituency import (
    ConstituencyMatch,
    assert_matches_constituency,
    get_constituency_matches,
    get_constituency_tree,
    match_constituency,
    parse_pattern,
)
from pytest_nlp.matchers import (
    MatchResult,
    assert_matches_dependency,
    assert_matches_phrases,
    assert_matches_tokens,
    match_dependency,
    match_phrases,
    match_tokens,
)
from pytest_nlp.models import (
    clear_model_cache,
    get_embedding_model,
    get_spacy_model,
    get_stanza_pipeline,
)
from pytest_nlp.plugin import NLPConfig, get_config
from pytest_nlp.semantic import (
    assert_semantic_contains,
    assert_semantic_similarity,
    semantic_contains,
    semantic_similarity,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Semantic similarity
    "semantic_similarity",
    "semantic_contains",
    "assert_semantic_contains",
    "assert_semantic_similarity",
    # SpaCy matchers
    "match_tokens",
    "match_phrases",
    "match_dependency",
    "assert_matches_tokens",
    "assert_matches_phrases",
    "assert_matches_dependency",
    "MatchResult",
    # Constituency parsing
    "match_constituency",
    "get_constituency_matches",
    "get_constituency_tree",
    "assert_matches_constituency",
    "parse_pattern",
    "ConstituencyMatch",
    # Model management
    "get_embedding_model",
    "get_spacy_model",
    "get_stanza_pipeline",
    "clear_model_cache",
    # Configuration
    "NLPConfig",
    "get_config",
]

