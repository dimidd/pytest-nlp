# pytest-nlp

A pytest plugin for evaluating free-text outputs using NLP techniques. Ideal for testing LLM responses, document processing pipelines, and other natural language applications.

## Features

- **Semantic Similarity**: Check if text is semantically similar using sentence-transformers
- **Token Matching**: Match token patterns using spaCy's Matcher
- **Phrase Matching**: Match exact phrases using spaCy's PhraseMatcher
- **Dependency Matching**: Match dependency graph patterns using spaCy's DependencyMatcher
- **Constituency Parsing**: Match phrase structure patterns using Stanza
- **AMR Parsing**: Match Abstract Meaning Representation patterns using amrlib (optional)

## Installation

```bash
pip install pytest-nlp
```

You'll also need to download the required models:

```bash
# spaCy model
python -m spacy download en_core_web_sm

# Stanza models are downloaded automatically on first use
```

## Quick Start

```python
from pytest_nlp import (
    assert_semantic_contains,
    assert_matches_dependency,
    assert_matches_constituency,
)

def test_drug_discontinuation():
    doc = """Patient was prescribed Ibuprofen on 01/12/2023 for headaches.
    Patient reported mild side effects. Ibuprofen was discontinued on 12/30/2023."""

    # Semantic matching: check if concept is present
    assert_semantic_contains(
        query="Ibuprofen was discontinued",
        document=doc,
        threshold=0.75,
    )

    # Check only last 2 sentences
    assert_semantic_contains(
        query="medication stopped",
        document=doc,
        sentences=[-2, -1],
        threshold=0.6,
    )
```

## Semantic Similarity

Use sentence-transformers to check semantic similarity:

```python
from pytest_nlp import semantic_similarity, semantic_contains, assert_semantic_contains

# Get similarity score between two texts
score = semantic_similarity("Hello world", "Hi there world")

# Check if query is semantically contained in document
is_contained, score = semantic_contains(
    query="The medication was stopped",
    document="Ibuprofen was discontinued on 12/30/2023.",
    threshold=0.6,
)

# Assert with custom model and parameters
assert_semantic_contains(
    query="drug discontinued",
    document=doc,
    model_name="all-MiniLM-L6-v2",
    metric="cosine",  # or "euclidean", "dot"
    threshold=0.7,
    truncate_dim=256,  # For matryoshka embeddings
)
```

### Sentence Filtering

Filter which sentences to search:

```python
# Check only last sentence
assert_semantic_contains(query, doc, sentences=[-1])

# Check first two sentences
assert_semantic_contains(query, doc, sentences=[0, 1])

# Check last 3 sentences using slice
assert_semantic_contains(query, doc, sentences=slice(-3, None))
```

## SpaCy Matchers

### Token Matching

```python
from pytest_nlp import match_tokens, assert_matches_tokens

# Match token patterns
results = match_tokens(
    doc="The quick brown fox jumps",
    patterns=[[{"LOWER": "quick"}, {"LOWER": "brown"}]],
)

# Assert pattern matches
assert_matches_tokens(
    doc="Patient was prescribed Ibuprofen",
    patterns=[[{"LEMMA": "prescribe"}]],
)
```

### Phrase Matching

```python
from pytest_nlp import match_phrases, assert_matches_phrases

# Match exact phrases
results = match_phrases(
    doc="The quick brown fox",
    phrases=["quick brown fox"],
)

# Case-insensitive matching
assert_matches_phrases(
    doc="Patient took IBUPROFEN",
    phrases=["ibuprofen"],
    attr="LOWER",
)
```

### Dependency Matching

```python
from pytest_nlp import match_dependency, assert_matches_dependency

# Match dependency patterns
assert_matches_dependency(
    doc="Ibuprofen was discontinued on 12/30/2023.",
    patterns=[
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "discontinue"}},
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "drug",
            "RIGHT_ATTRS": {"LOWER": "ibuprofen", "DEP": "nsubjpass"},
        },
    ],
)
```

## Constituency Parsing

Match phrase structure patterns using Stanza:

```python
from pytest_nlp import match_constituency, assert_matches_constituency, get_constituency_tree

# Get constituency tree (useful for debugging)
trees = get_constituency_tree("Ibuprofen was discontinued.")
print(trees[0])
# (S (NP (NNP Ibuprofen)) (VP (VBD was) (VP (VBN discontinued))))

# Match with variable capture
matches = match_constituency(
    doc="Ibuprofen was discontinued on 12/30/2023.",
    pattern="(S (NP (NNP ?drug)) (VP (VBD was) (VP (VBN discontinued) ...)))",
)
print(matches[0]["drug"])  # "Ibuprofen"

# Assert pattern matches
assert_matches_constituency(
    doc="The cat sat on the mat.",
    pattern="(NP (DT ?det) (NN ?noun))",
)
```

### Pattern Syntax

- **Exact matching**: `(NP (NNP Ibuprofen))` - matches specific structure
- **Variable capture**: `(NNP ?drug)` - captures text as variable
- **Partial matching**: `(VP ...)` or `(VP (VBD was) ...)` - matches with additional children
- **POS wildcards**: `(VB*)` - matches VB, VBD, VBN, VBZ, etc.

## AMR (Abstract Meaning Representation)

AMR provides deep semantic parsing. Install the optional dependency:

```bash
pip install pytest-nlp[amr]
```

### Parsing Sentences to AMR

```python
from pytest_nlp import parse_amr, assert_has_concept, assert_has_role

# Parse a sentence to AMR
graph = parse_amr("The boy wants to go.")
print(graph)
# (w / want-01
#    :ARG0 (b / boy)
#    :ARG1 (g / go-02
#       :ARG0 b))

# Check for specific concepts
assert graph.has_concept("want-01")
assert graph.has_concept("boy")
```

### Concept and Role Assertions

```python
# Assert concept exists
assert_has_concept("The doctor discontinued the medication.", "discontinue-01")

# Assert role exists
assert_has_role(
    "The boy wants to go.",
    role=":ARG0",
    source_concept="want-01",
)
```

### Negation Detection

```python
from pytest_nlp import assert_is_negated, assert_not_negated

# Check for negation
assert_is_negated("The boy does not want to go.")
assert_not_negated("The boy wants to go.")

# Check specific concept negation
assert_is_negated("The doctor did not discontinue the medication.", concept="discontinue-01")
```

### Pattern Matching

```python
from pytest_nlp import match_amr_pattern, assert_amr_pattern, find_concepts

# Find all instances of a concept
matches = find_concepts(graph, "want-*")  # Wildcard support

# Match patterns with specific roles
assert_amr_pattern(
    "The boy wants to go.",
    concept="want-01",
    roles={":ARG0": "boy", ":ARG1": "*"},  # * matches anything
)
```

### Semantic Similarity

Compare semantic structure between sentences:

```python
from pytest_nlp import sentence_amr_similarity, assert_amr_similarity

# Get Smatch F1 score
score = sentence_amr_similarity(
    "The boy wants to go.",
    "The child wants to leave.",
)

# Assert sentences have similar AMR structure
assert_amr_similarity(
    "The boy wants to go.",
    "The child wants to leave.",
    threshold=0.5,
)
```

## Configuration

Configure defaults in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
nlp_embedding_model = "all-MiniLM-L6-v2"
nlp_spacy_model = "en_core_web_sm"
nlp_stanza_lang = "en"
nlp_similarity_threshold = "0.7"
nlp_similarity_metric = "cosine"
```

All options can be overridden via function parameters.

## Model Management

Models are cached per-session for performance:

```python
from pytest_nlp import get_embedding_model, get_spacy_model, get_stanza_pipeline, clear_model_cache

# Get cached models
model = get_embedding_model("all-MiniLM-L6-v2")
nlp = get_spacy_model("en_core_web_sm")
pipeline = get_stanza_pipeline("en")

# Clear cache if needed
clear_model_cache()
```

## License

MIT

