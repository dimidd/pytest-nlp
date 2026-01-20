<!-- 
  README TEMPLATE - Do not edit README.md directly!
  Edit this file and run: npx markdown-autodocs -c code-block -o ./README.md
-->

# pytest-nlp

A pytest plugin for evaluating free-text outputs using NLP techniques. Ideal for testing LLM responses, document processing pipelines, and other natural language applications.

## Features

- **Semantic Similarity**: Check if text is semantically similar using sentence-transformers
- **Token Matching**: Match token patterns using spaCy's Matcher
- **Phrase Matching**: Match exact phrases using spaCy's PhraseMatcher
- **Dependency Matching**: Match dependency graph patterns using spaCy's DependencyMatcher
- **Constituency Parsing**: Match phrase structure patterns using Stanza
- **AMR Parsing**: Match Abstract Meaning Representation patterns using amrlib (optional)
- **Medical NLP**: Clinical entity extraction and medical text analysis — [see Medical README](MEDICAL_README.md)

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

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=34-52) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_semantic_contains

    doc = """The new smartphone features a 6.5-inch display.
    It has an excellent camera system. Battery life is outstanding."""

    # Semantic matching: check if concept is present
    assert_semantic_contains(
        query="great battery performance",
        document=doc,
        threshold=0.6,
    )

    # Check only last sentence
    assert_semantic_contains(
        query="long battery life",
        document=doc,
        sentences=[-1],
        threshold=0.5,
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

## Semantic Similarity

Use sentence-transformers to check semantic similarity:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=62-67) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import semantic_similarity

    # Get similarity score between two texts
    score = semantic_similarity("Hello world", "Hi there world")
    assert score > 0.5
```
<!-- MARKDOWN-AUTO-DOCS:END -->

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=71-82) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import semantic_contains

    # Check if query is semantically contained in document
    is_contained, score = semantic_contains(
        query="The project was completed",
        document="We finished the project on Friday.",
        threshold=0.6,
    )
    assert is_contained
    assert score >= 0.6
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Sentence Filtering

Filter which sentences to search:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=90-103) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_semantic_contains

    doc = "First sentence. Second sentence. Third sentence. Last sentence."

    # Check only last sentence
    assert_semantic_contains("Last sentence", doc, sentences=[-1], threshold=0.5)

    # Check first two sentences
    assert_semantic_contains("First", doc, sentences=[0, 1], threshold=0.3)

    # Check last 3 sentences using slice
    assert_semantic_contains("Last", doc, sentences=slice(-3, None), threshold=0.3)
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Pattern-Based Sentence Selection

Use `SentenceMatch` to select sentences containing a word or matching a regex:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=111-144) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    import re

    from pytest_nlp import SentenceMatch, assert_semantic_contains

    doc = """The order was placed on Monday.
    Processing took 2 business days.
    Shipping cost was $15.99.
    The order arrived on Thursday."""

    # Select first sentence containing "order" (case-insensitive)
    assert_semantic_contains(
        query="order was placed",
        document=doc,
        sentences=SentenceMatch("order", mode="first"),
        threshold=0.5,
    )

    # Select last sentence containing "order"
    assert_semantic_contains(
        query="order arrived",
        document=doc,
        sentences=SentenceMatch("order", mode="last"),
        threshold=0.5,
    )

    # Select all sentences matching a regex pattern (e.g., containing prices)
    assert_semantic_contains(
        query="shipping cost",
        document=doc,
        sentences=SentenceMatch(re.compile(r"\$\d+"), mode="all"),
        threshold=0.3,
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

The `mode` parameter controls which matching sentences to use:
- `"first"`: Only the first matching sentence (default)
- `"last"`: Only the last matching sentence
- `"all"`: All matching sentences

## SpaCy Matchers

### Token Matching

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=152-166) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_matches_tokens, match_tokens

    # Match token patterns
    results = match_tokens(
        doc="The quick brown fox jumps over the lazy dog",
        patterns=[[{"LOWER": "quick"}, {"LOWER": "brown"}]],
    )
    assert len(results) >= 1
    assert results[0].text == "quick brown"

    # Assert pattern matches
    assert_matches_tokens(
        doc="The company announced record profits",
        patterns=[[{"LEMMA": "announce"}]],
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Phrase Matching

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=176-191) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_matches_phrases, match_phrases

    # Match exact phrases
    results = match_phrases(
        doc="The quick brown fox",
        phrases=["quick brown fox"],
    )
    assert len(results) >= 1
    assert results[0].text == "quick brown fox"

    # Case-insensitive matching
    assert_matches_phrases(
        doc="Order shipped via FEDEX",
        phrases=["fedex"],
        attr="LOWER",
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Dependency Matching

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=201-220) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_matches_dependency

    # Match dependency patterns (subject-verb-object)
    assert_matches_dependency(
        doc="The engineer fixed the bug yesterday.",
        patterns=[
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "fix"}},
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": "nsubj"},
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {"DEP": "dobj"},
            },
        ],
```
<!-- MARKDOWN-AUTO-DOCS:END -->

## Constituency Parsing

Match phrase structure patterns using Stanza:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=231-254) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import (
        assert_matches_constituency,
        get_constituency_tree,
        match_constituency,
    )

    # Get constituency tree (useful for debugging)
    trees = get_constituency_tree("The cat sat on the mat.")
    assert len(trees) == 1
    assert trees[0].startswith("(")

    # Match with variable capture
    matches = match_constituency(
        doc="The engineer fixed the bug.",
        pattern="(S (NP (DT ?det) (NN ?subject)) ...)",
    )
    assert len(matches) >= 1
    assert matches[0]["subject"] == "engineer"

    # Assert pattern matches
    assert_matches_constituency(
        doc="The cat sat on the mat.",
        pattern="(NP (DT ?det) (NN ?noun))",
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Pattern Syntax

- **Exact matching**: `(NP (DT the) (NN cat))` - matches specific structure
- **Variable capture**: `(NN ?noun)` - captures text as variable
- **Partial matching**: `(VP ...)` or `(VP (VBD sat) ...)` - matches with additional children
- **POS wildcards**: `(VB*)` - matches VB, VBD, VBN, VBZ, etc.

## AMR (Abstract Meaning Representation)

AMR provides deep semantic parsing. Install the optional dependency:

```bash
pip install pytest-nlp[amr]
```

### Parsing Sentences to AMR

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=270-285) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_has_concept, assert_has_role, parse_amr

    # Parse a sentence to AMR
    graph = parse_amr("The boy wants to go.")
    assert graph.has_concept("want-01")
    assert graph.has_concept("boy")

    # Assert concept exists
    assert_has_concept("The manager approved the request.", "approve-01")

    # Assert role exists
    assert_has_role(
        "The boy wants to go.",
        role=":ARG0",
        source_concept="want-01",
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Negation Detection

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=290-300) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_is_negated, assert_not_negated

    # Check for negation
    assert_is_negated("The boy does not want to go.")
    assert_not_negated("The boy wants to go.")

    # Check specific concept negation
    assert_is_negated(
        "The manager did not approve the request.",
        concept="approve-01",
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Pattern Matching

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=305-318) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_amr_pattern, find_concepts, parse_amr

    graph = parse_amr("The boy wants to go.")

    # Find all instances of a concept
    matches = find_concepts(graph, "want-*")  # Wildcard support
    assert len(matches) >= 1

    # Match patterns with specific roles
    assert_amr_pattern(
        "The boy wants to go.",
        concept="want-01",
        roles={":ARG0": "boy", ":ARG1": "*"},  # * matches anything
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Semantic Similarity

Compare semantic structure between sentences:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_readme_examples.py&lines=323-338) -->
<!-- The below code snippet is automatically added from ./tests/test_readme_examples.py -->
```py
    from pytest_nlp import assert_amr_similarity, sentence_amr_similarity

    # Get Smatch F1 score
    score = sentence_amr_similarity(
        "The boy wants to go.",
        "The child wants to leave.",
    )
    assert score > 0

    # Assert sentences have similar AMR structure
    assert_amr_similarity(
        "The boy wants to go.",
        "The child wants to leave.",
        threshold=0.5,
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

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

## Medical NLP

For clinical and healthcare applications, pytest-nlp provides specialized tools including medical entity extraction using MedSpaCy. See the **[Medical README](MEDICAL_README.md)** for:

- Medical NER (drugs, problems, diagnoses)
- Clinical negation detection
- Medical-focused examples of all features above

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
