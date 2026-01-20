<!-- 
  MEDICAL README TEMPLATE - Do not edit MEDICAL_README.md directly!
  Edit this file and run: npx markdown-autodocs -c code-block -o ./MEDICAL_README.md
-->

# Medical NLP with pytest-nlp

This guide provides medical and clinical examples for all pytest-nlp features, plus documentation for the MedSpaCy-based medical NER module.

## Installation

For medical NER functionality, install with the `medical` extra:

```bash
pip install pytest-nlp[medical]
```

This installs [MedSpaCy](https://github.com/medspacy/medspacy) and its dependencies.

For AMR functionality with clinical text:

```bash
pip install pytest-nlp[amr]
```

## Quick Start

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=24-39) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_has_drug, assert_semantic_contains

    note = """Patient was prescribed Ibuprofen on 01/12/2023 for headaches.
    Patient reported mild side effects. Ibuprofen was discontinued on 12/30/2023."""

    # Semantic: check if discontinuation is mentioned
    assert_semantic_contains(
        query="Ibuprofen was discontinued",
        document=note,
        threshold=0.7,
    )

    # Medical NER: verify drug is mentioned
    assert_has_drug(note, drug_name="ibuprofen")
```
<!-- MARKDOWN-AUTO-DOCS:END -->

---

## Semantic Similarity for Clinical Text

Use semantic similarity to verify clinical concepts are present:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=47-62) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_semantic_contains, semantic_contains

    # Check if clinical concept is semantically present
    is_contained, score = semantic_contains(
        query="drug discontinued",
        document="Ibuprofen was discontinued on 12/30/2023.",
        threshold=0.5,
    )
    assert is_contained

    # Verify diagnosis mentioned
    assert_semantic_contains(
        query="patient has high blood pressure",
        document="Assessment: Hypertension, well-controlled on current medications.",
        threshold=0.4,
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Filtering Clinical Sections

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=67-80) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import SentenceMatch, assert_semantic_contains

    discharge_summary = """Chief Complaint: Chest pain for 2 days.
    History: Patient is a 65-year-old male with history of diabetes.
    Assessment: Acute coronary syndrome ruled out.
    Plan: Continue current medications. Follow up in 2 weeks."""

    # Check the Plan section
    assert_semantic_contains(
        query="maintain current treatment",
        document=discharge_summary,
        sentences=SentenceMatch("Plan", mode="first"),
        threshold=0.4,
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Searching for Drug Mentions

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=85-107) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import SentenceMatch, assert_semantic_contains

    note = """Patient was prescribed Ibuprofen 200mg on 01/12/2023.
    Patient reported mild side effects including nausea.
    Dosage increased to 400mg on 02/15/2023.
    Ibuprofen was discontinued on 12/30/2023 due to GI symptoms."""

    # Find first sentence about Ibuprofen
    assert_semantic_contains(
        query="prescribed Ibuprofen",
        document=note,
        sentences=SentenceMatch("ibuprofen", mode="first"),
        threshold=0.5,
    )

    # Find last sentence about Ibuprofen (likely discontinuation)
    assert_semantic_contains(
        query="Ibuprofen discontinued",
        document=note,
        sentences=SentenceMatch("ibuprofen", mode="last"),
        threshold=0.5,
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

---

## SpaCy Matchers for Clinical Text

### Token Matching

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=116-129) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_matches_tokens, match_tokens

    # Match medication administration patterns
    assert_matches_tokens(
        doc="Patient was prescribed Metformin 500mg twice daily.",
        patterns=[[{"LEMMA": "prescribe"}]],
    )

    # Match symptom patterns
    results = match_tokens(
        doc="Patient reports severe chest pain radiating to left arm.",
        patterns=[[{"LOWER": "chest"}, {"LOWER": "pain"}]],
    )
    assert len(results) >= 1
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Phrase Matching

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=134-147) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_matches_phrases

    # Match clinical phrases
    assert_matches_phrases(
        doc="Patient presents with shortness of breath and chest pain.",
        phrases=["shortness of breath", "chest pain"],
    )

    # Case-insensitive drug matching
    assert_matches_phrases(
        doc="Started METFORMIN 500mg BID",
        phrases=["metformin"],
        attr="LOWER",
    )
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Dependency Matching

Match clinical relationships using dependency patterns:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=152-166) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_matches_dependency

    # Match drug discontinuation pattern
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
<!-- MARKDOWN-AUTO-DOCS:END -->

---

## Constituency Parsing for Clinical Text

Match phrase structure patterns in clinical notes:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=176-190) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import get_constituency_tree, match_constituency

    # Get constituency tree for clinical sentence
    trees = get_constituency_tree("Ibuprofen was discontinued.")
    assert len(trees) == 1
    assert "discontinued" in trees[0]

    # Match drug discontinuation with variable capture
    # Use partial matching (...) at S level to account for punctuation
    matches = match_constituency(
        doc="Ibuprofen was discontinued on 12/30/2023.",
        pattern="(S (NP (NNP ?drug)) ...)",
    )
    assert len(matches) >= 1
    assert matches[0]["drug"] == "Ibuprofen"
```
<!-- MARKDOWN-AUTO-DOCS:END -->

---

## Medical NER (MedSpaCy)

The medical NER module uses [MedSpaCy](https://github.com/medspacy/medspacy) for clinical entity extraction.

### Extracting Medical Entities

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=200-210) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import MedicalEntity, extract_medical_entities

    note = """Patient takes ibuprofen for headaches. No fever reported.
    Possible pneumonia. History of diabetes."""

    entities = extract_medical_entities(note)
    assert len(entities) > 0

    # Get only drugs
    drugs = extract_medical_entities(note, labels=["DRUG"])
    assert any(e.text.lower() == "ibuprofen" for e in drugs)
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### The MedicalEntity Dataclass

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The entity text |
| `label` | `str` | Entity type (e.g., 'DRUG', 'PROBLEM') |
| `start` | `int` | Start character offset |
| `end` | `int` | End character offset |
| `is_negated` | `bool` | Whether entity is negated (e.g., "no fever") |
| `is_uncertain` | `bool` | Whether entity is uncertain (e.g., "possible pneumonia") |
| `is_historical` | `bool` | Whether entity is historical (e.g., "history of diabetes") |
| `is_family` | `bool` | Whether entity refers to family history |
| `section` | `str \| None` | Clinical section containing the entity |

### Assertion Functions

#### assert_has_drug

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=215-225) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_has_drug

    # Assert any drug is mentioned
    assert_has_drug("Patient takes ibuprofen daily.")

    # Assert specific drug
    assert_has_drug("Patient takes ibuprofen daily.", drug_name="ibuprofen")

    # Assert drug is NOT negated (affirmed)
    assert_has_drug("Patient takes ibuprofen daily.", negated=False)
```
<!-- MARKDOWN-AUTO-DOCS:END -->

#### assert_has_problem

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=229-236) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_has_problem

    # Assert any problem is mentioned
    assert_has_problem("Patient presents with fever.")

    # Assert specific problem
    assert_has_problem("Patient presents with fever.", problem="fever")
```
<!-- MARKDOWN-AUTO-DOCS:END -->

#### assert_entity_negated / assert_entity_not_negated

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=240-249) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_entity_negated, assert_entity_not_negated

    note = "Patient denies chest pain. Reports headache."

    # Assert chest pain is negated
    assert_entity_negated(note, entity_text="chest pain")

    # Assert headache is NOT negated
    assert_entity_not_negated(note, entity_text="headache")
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Supported Entity Types

| Label | Description | Examples |
|-------|-------------|----------|
| `DRUG` | Medications | ibuprofen, aspirin, metformin |
| `PROBLEM` | Diagnoses/Symptoms | diabetes, fever, chest pain |

### Negation and Context Detection

MedSpaCy's ConText algorithm detects contextual modifiers:

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=253-266) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import extract_medical_entities

    note = """
    Patient denies chest pain.
    No fever reported.
    Possible pneumonia.
    History of diabetes.
    """

    entities = extract_medical_entities(note)

    # Check negation
    negated = [e for e in entities if e.is_negated]
    assert len(negated) >= 1
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Custom Pipeline

```python
from pytest_nlp import get_medspacy_pipeline, extract_medical_entities
from medspacy.target_matcher import TargetRule

# Get and customize the pipeline
nlp = get_medspacy_pipeline()
target_matcher = nlp.get_pipe("medspacy_target_matcher")
target_matcher.add([
    TargetRule("myocardial infarction", "PROBLEM"),
    TargetRule("warfarin", "DRUG"),
    TargetRule("ECG", "TEST"),
])

# Use the custom pipeline
entities = extract_medical_entities(
    "Patient had myocardial infarction, started on warfarin.",
    pipeline=nlp,
)
```

---

## AMR for Clinical Text

AMR (Abstract Meaning Representation) provides deep semantic parsing useful for clinical NLP:

```bash
pip install pytest-nlp[amr]
```

### Parsing Clinical Sentences

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=282-290) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_has_concept, parse_amr

    # Parse clinical action
    graph = parse_amr("The doctor discontinued the medication.")
    assert graph.has_concept("discontinue-01")
    assert graph.has_concept("doctor")

    # Assert discontinuation concept exists
    assert_has_concept("The doctor discontinued Ibuprofen.", "discontinue-01")
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Clinical Negation Detection

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=295-300) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_is_negated, assert_not_negated

    # Negated clinical findings (using explicit negation)
    assert_is_negated("The doctor did not discontinue the medication.")
    assert_not_negated("The doctor discontinued the medication.")
```
<!-- MARKDOWN-AUTO-DOCS:END -->

### Comparing Clinical Sentences

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./tests/test_medical_readme_examples.py&lines=304-313) -->
<!-- The below code snippet is automatically added from ./tests/test_medical_readme_examples.py -->
```py
    from pytest_nlp import assert_amr_similarity, sentence_amr_similarity

    # Compare semantically similar clinical statements
    score = sentence_amr_similarity(
        "The doctor discontinued the medication.",
        "The physician stopped the drug.",
    )
    assert score > 0
```
<!-- MARKDOWN-AUTO-DOCS:END -->

---

## API Reference

### Medical NER Functions

| Function | Description |
|----------|-------------|
| `extract_medical_entities(doc, labels, include_negated, pipeline)` | Extract medical entities from text |
| `assert_has_medical_entity(doc, entity_type, text, negated, min_count, pipeline, msg)` | Assert entity exists |
| `assert_has_drug(doc, drug_name, negated, pipeline, msg)` | Assert drug entity exists |
| `assert_has_problem(doc, problem, negated, pipeline, msg)` | Assert problem entity exists |
| `assert_entity_negated(doc, entity_text, entity_type, pipeline, msg)` | Assert entity is negated |
| `assert_entity_not_negated(doc, entity_text, entity_type, pipeline, msg)` | Assert entity is NOT negated |
| `get_medspacy_pipeline(enable, load_rules)` | Get cached MedSpaCy pipeline |
| `clear_medspacy_cache()` | Clear pipeline cache |

### Classes

| Class | Description |
|-------|-------------|
| `MedicalEntity` | Dataclass representing an extracted medical entity |
