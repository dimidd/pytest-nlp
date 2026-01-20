"""pytest-nlp: A pytest plugin for evaluating free-text outputs using NLP.

This plugin provides assertions for:
- Semantic similarity using sentence transformers
- Token, phrase, and dependency matching using spaCy
- Constituency parsing patterns using Stanza
- Medical NER using MedSpaCy (optional)
- AMR (Abstract Meaning Representation) using amrlib (optional)
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
    ANY,
    DEFAULT_AMR_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SPACY_MODEL,
    DEFAULT_STANZA_LANG,
    DEFAULT_STANZA_PROCESSORS,
    SentenceMatch,
    clear_model_cache,
    get_embedding_model,
    get_spacy_model,
    get_stanza_pipeline,
    split_sentences,
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
    # Sentence utilities
    "ANY",
    "SentenceMatch",
    "split_sentences",
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
    # Model defaults
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SPACY_MODEL",
    "DEFAULT_STANZA_LANG",
    "DEFAULT_STANZA_PROCESSORS",
    "DEFAULT_AMR_MODEL",
    # Configuration
    "NLPConfig",
    "get_config",
]

# Medical NER (optional, requires medspacy)
try:
    from pytest_nlp.medical import (
        MedicalEntity,
        assert_entity_negated,
        assert_entity_not_negated,
        assert_has_drug,
        assert_has_medical_entity,
        assert_has_problem,
        clear_medspacy_cache,
        extract_medical_entities,
        get_medspacy_pipeline,
    )

    __all__.extend([
        # Medical NER
        "MedicalEntity",
        "extract_medical_entities",
        "assert_has_medical_entity",
        "assert_has_drug",
        "assert_has_problem",
        "assert_entity_negated",
        "assert_entity_not_negated",
        "get_medspacy_pipeline",
        "clear_medspacy_cache",
    ])
except ImportError:
    # medspacy not installed, medical functions not available
    pass

# AMR (optional, requires amrlib and penman)
try:
    from pytest_nlp.amr import (
        AMRGraph,
        AMRMatch,
        assert_amr_pattern,
        assert_amr_similarity,
        assert_has_concept,
        assert_has_role,
        assert_is_negated,
        assert_not_negated,
        clear_amr_cache,
        download_amr_model,
        find_concepts,
        get_amr_parser,
        match_amr_pattern,
        parse_amr,
        parse_amr_batch,
        sentence_amr_similarity,
    )

    __all__.extend([
        # AMR
        "AMRGraph",
        "AMRMatch",
        "parse_amr",
        "parse_amr_batch",
        "find_concepts",
        "match_amr_pattern",
        "sentence_amr_similarity",
        "assert_has_concept",
        "assert_has_role",
        "assert_is_negated",
        "assert_not_negated",
        "assert_amr_pattern",
        "assert_amr_similarity",
        "get_amr_parser",
        "download_amr_model",
        "clear_amr_cache",
    ])
except ImportError:
    # amrlib not installed, AMR functions not available
    pass

