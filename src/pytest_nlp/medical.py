"""Medical NER module using MedSpaCy.

This module provides functions for extracting and asserting medical entities
using MedSpaCy's clinical NLP pipeline, including:
- Drug/medication detection
- Problem/diagnosis detection  
- Negation detection (ConText)
- Section detection
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import spacy


EntityLabel = Literal["DRUG", "PROBLEM", "TREATMENT", "TEST", "ANATOMICAL_SITE"]


@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text.

    :param text: The entity text.
    :param label: Entity type label (e.g., 'DRUG', 'PROBLEM').
    :param start: Start character offset.
    :param end: End character offset.
    :param is_negated: Whether the entity is negated (e.g., "no fever").
    :param is_uncertain: Whether the entity is uncertain (e.g., "possible pneumonia").
    :param is_historical: Whether the entity is historical (e.g., "history of diabetes").
    :param is_family: Whether the entity refers to family history.
    :param section: The clinical section containing this entity (if detected).
    """

    text: str
    label: str
    start: int
    end: int
    is_negated: bool = False
    is_uncertain: bool = False
    is_historical: bool = False
    is_family: bool = False
    section: str | None = None


@lru_cache(maxsize=4)
def get_medspacy_pipeline(
    enable: tuple[str, ...] = ("medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"),
    load_rules: bool = True,
) -> "spacy.Language":
    """Load and cache a MedSpaCy pipeline.

    :param enable: Tuple of components to enable.
    :param load_rules: Whether to load default clinical rules.
    :returns: The loaded MedSpaCy pipeline.

    >>> nlp = get_medspacy_pipeline()
    >>> doc = nlp("Patient has diabetes.")
    >>> len(doc.ents) >= 0
    True
    """
    try:
        import medspacy
    except ImportError as e:
        raise ImportError(
            "medspacy is required for medical NER functions. "
            "Install it with: pip install pytest-nlp[medical]"
        ) from e

    nlp = medspacy.load(enable=list(enable))

    if load_rules:
        # Add default target rules for common medical entities
        from medspacy.target_matcher import TargetRule

        target_matcher = nlp.get_pipe("medspacy_target_matcher")

        # Common drug patterns
        drug_rules = [
            TargetRule("ibuprofen", "DRUG"),
            TargetRule("aspirin", "DRUG"),
            TargetRule("acetaminophen", "DRUG"),
            TargetRule("metformin", "DRUG"),
            TargetRule("lisinopril", "DRUG"),
            TargetRule("amoxicillin", "DRUG"),
            TargetRule("omeprazole", "DRUG"),
            TargetRule("prednisone", "DRUG"),
            TargetRule("hydrocodone", "DRUG"),
            TargetRule("gabapentin", "DRUG"),
        ]

        # Common problem/diagnosis patterns
        problem_rules = [
            TargetRule("diabetes", "PROBLEM"),
            TargetRule("hypertension", "PROBLEM"),
            TargetRule("pneumonia", "PROBLEM"),
            TargetRule("fever", "PROBLEM"),
            TargetRule("headache", "PROBLEM"),
            TargetRule("chest pain", "PROBLEM"),
            TargetRule("shortness of breath", "PROBLEM"),
            TargetRule("cough", "PROBLEM"),
            TargetRule("nausea", "PROBLEM"),
            TargetRule("infection", "PROBLEM"),
        ]

        target_matcher.add(drug_rules)
        target_matcher.add(problem_rules)

    return nlp


def _get_entity_modifiers(ent) -> dict[str, bool]:
    """Extract modifier attributes from a MedSpaCy entity.

    :param ent: spaCy entity with MedSpaCy extensions.
    :returns: Dictionary of modifier flags.
    """
    modifiers = {
        "is_negated": False,
        "is_uncertain": False,
        "is_historical": False,
        "is_family": False,
    }

    # Check if entity has context modifiers
    if hasattr(ent._, "context_attributes"):
        attrs = ent._.context_attributes
        modifiers["is_negated"] = attrs.get("is_negated", False)
        modifiers["is_uncertain"] = attrs.get("is_uncertain", False)
        modifiers["is_historical"] = attrs.get("is_historical", False)
        modifiers["is_family"] = attrs.get("is_family", False)
    elif hasattr(ent._, "is_negated"):
        # Direct attribute access
        modifiers["is_negated"] = getattr(ent._, "is_negated", False)
        modifiers["is_uncertain"] = getattr(ent._, "is_uncertain", False)
        modifiers["is_historical"] = getattr(ent._, "is_historical", False)
        modifiers["is_family"] = getattr(ent._, "is_family", False)

    return modifiers


def extract_medical_entities(
    doc: str,
    labels: list[str] | None = None,
    include_negated: bool = True,
    pipeline: "spacy.Language | None" = None,
) -> list[MedicalEntity]:
    """Extract medical entities from text using MedSpaCy.

    :param doc: Document text to process.
    :param labels: Filter to specific entity labels (e.g., ['DRUG', 'PROBLEM']).
    :param include_negated: Whether to include negated entities.
    :param pipeline: Optional custom MedSpaCy pipeline.
    :returns: List of MedicalEntity objects.

    >>> entities = extract_medical_entities("Patient takes ibuprofen for headache.")
    >>> any(e.label == "DRUG" for e in entities)
    True
    """
    nlp = pipeline or get_medspacy_pipeline()
    spacy_doc = nlp(doc)

    entities = []
    for ent in spacy_doc.ents:
        # Filter by label if specified
        if labels and ent.label_ not in labels:
            continue

        modifiers = _get_entity_modifiers(ent)

        # Skip negated if not wanted
        if not include_negated and modifiers["is_negated"]:
            continue

        # Get section if available
        section = None
        if hasattr(ent._, "section_category"):
            section = ent._.section_category

        entities.append(
            MedicalEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                section=section,
                **modifiers,
            )
        )

    return entities


def assert_has_medical_entity(
    doc: str,
    entity_type: str,
    text: str | None = None,
    negated: bool | None = None,
    min_count: int = 1,
    pipeline: "spacy.Language | None" = None,
    msg: str | None = None,
) -> None:
    """Assert that a medical entity of a specific type exists in the document.

    :param doc: Document text to process.
    :param entity_type: Entity type to look for (e.g., 'DRUG', 'PROBLEM').
    :param text: Optional specific text the entity should match (case-insensitive).
    :param negated: If specified, filter by negation status.
    :param min_count: Minimum number of matching entities required.
    :param pipeline: Optional custom MedSpaCy pipeline.
    :param msg: Custom error message.
    :raises AssertionError: If no matching entity is found.
    """
    entities = extract_medical_entities(doc, labels=[entity_type], pipeline=pipeline)

    # Filter by text if specified
    if text:
        entities = [e for e in entities if e.text.lower() == text.lower()]

    # Filter by negation if specified
    if negated is not None:
        entities = [e for e in entities if e.is_negated == negated]

    if len(entities) < min_count:
        if msg:
            raise AssertionError(msg)

        all_entities = extract_medical_entities(doc, pipeline=pipeline)
        entity_summary = ", ".join(f"{e.text} ({e.label})" for e in all_entities) or "none"

        raise AssertionError(
            f"Expected at least {min_count} {entity_type} entity(ies)"
            f"{f' with text {text!r}' if text else ''}"
            f"{f' (negated={negated})' if negated is not None else ''}, "
            f"found {len(entities)}.\n"
            f"  Document: {doc!r}\n"
            f"  All entities found: {entity_summary}"
        )


def assert_has_drug(
    doc: str,
    drug_name: str | None = None,
    negated: bool | None = None,
    pipeline: "spacy.Language | None" = None,
    msg: str | None = None,
) -> None:
    """Assert that a drug/medication entity exists in the document.

    :param doc: Document text to process.
    :param drug_name: Optional specific drug name to match.
    :param negated: If specified, filter by negation status.
    :param pipeline: Optional custom MedSpaCy pipeline.
    :param msg: Custom error message.
    :raises AssertionError: If no matching drug entity is found.
    """
    assert_has_medical_entity(
        doc=doc,
        entity_type="DRUG",
        text=drug_name,
        negated=negated,
        pipeline=pipeline,
        msg=msg,
    )


def assert_has_problem(
    doc: str,
    problem: str | None = None,
    negated: bool | None = None,
    pipeline: "spacy.Language | None" = None,
    msg: str | None = None,
) -> None:
    """Assert that a problem/diagnosis entity exists in the document.

    :param doc: Document text to process.
    :param problem: Optional specific problem text to match.
    :param negated: If specified, filter by negation status.
    :param pipeline: Optional custom MedSpaCy pipeline.
    :param msg: Custom error message.
    :raises AssertionError: If no matching problem entity is found.
    """
    assert_has_medical_entity(
        doc=doc,
        entity_type="PROBLEM",
        text=problem,
        negated=negated,
        pipeline=pipeline,
        msg=msg,
    )


def assert_entity_negated(
    doc: str,
    entity_text: str,
    entity_type: str | None = None,
    pipeline: "spacy.Language | None" = None,
    msg: str | None = None,
) -> None:
    """Assert that a specific entity is negated in the document.

    :param doc: Document text to process.
    :param entity_text: Text of the entity to check.
    :param entity_type: Optional entity type filter.
    :param pipeline: Optional custom MedSpaCy pipeline.
    :param msg: Custom error message.
    :raises AssertionError: If entity is not found or not negated.
    """
    labels = [entity_type] if entity_type else None
    entities = extract_medical_entities(doc, labels=labels, pipeline=pipeline)

    matching = [e for e in entities if e.text.lower() == entity_text.lower()]

    if not matching:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Entity {entity_text!r} not found in document.\n"
            f"  Document: {doc!r}"
        )

    negated = [e for e in matching if e.is_negated]
    if not negated:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Entity {entity_text!r} was found but is not negated.\n"
            f"  Document: {doc!r}"
        )


def assert_entity_not_negated(
    doc: str,
    entity_text: str,
    entity_type: str | None = None,
    pipeline: "spacy.Language | None" = None,
    msg: str | None = None,
) -> None:
    """Assert that a specific entity is NOT negated in the document.

    :param doc: Document text to process.
    :param entity_text: Text of the entity to check.
    :param entity_type: Optional entity type filter.
    :param pipeline: Optional custom MedSpaCy pipeline.
    :param msg: Custom error message.
    :raises AssertionError: If entity is not found or is negated.
    """
    labels = [entity_type] if entity_type else None
    entities = extract_medical_entities(doc, labels=labels, pipeline=pipeline)

    matching = [e for e in entities if e.text.lower() == entity_text.lower()]

    if not matching:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Entity {entity_text!r} not found in document.\n"
            f"  Document: {doc!r}"
        )

    not_negated = [e for e in matching if not e.is_negated]
    if not not_negated:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Entity {entity_text!r} was found but is negated.\n"
            f"  Document: {doc!r}"
        )


def clear_medspacy_cache() -> None:
    """Clear the cached MedSpaCy pipeline."""
    get_medspacy_pipeline.cache_clear()

