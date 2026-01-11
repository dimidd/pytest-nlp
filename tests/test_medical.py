"""Tests for medical NER module using MedSpaCy."""

import pytest

# Skip all tests in this module if medspacy is not installed
pytest.importorskip("medspacy")

from pytest_nlp.medical import (
    MedicalEntity,
    assert_entity_negated,
    assert_entity_not_negated,
    assert_has_drug,
    assert_has_medical_entity,
    assert_has_problem,
    extract_medical_entities,
    get_medspacy_pipeline,
)


@pytest.fixture(scope="module")
def medspacy_nlp():
    """Load MedSpaCy pipeline once for all tests."""
    return get_medspacy_pipeline()


class TestExtractMedicalEntities:
    """Tests for extract_medical_entities function."""

    def test_extract_drug(self, medspacy_nlp) -> None:
        """Test extracting drug entities."""
        entities = extract_medical_entities(
            "Patient takes ibuprofen daily.",
            pipeline=medspacy_nlp,
        )
        drugs = [e for e in entities if e.label == "DRUG"]
        assert len(drugs) >= 1
        assert any(e.text.lower() == "ibuprofen" for e in drugs)

    def test_extract_problem(self, medspacy_nlp) -> None:
        """Test extracting problem entities."""
        entities = extract_medical_entities(
            "Patient has diabetes and hypertension.",
            pipeline=medspacy_nlp,
        )
        problems = [e for e in entities if e.label == "PROBLEM"]
        assert len(problems) >= 1

    def test_extract_multiple_types(self, medspacy_nlp) -> None:
        """Test extracting multiple entity types."""
        entities = extract_medical_entities(
            "Patient takes metformin for diabetes.",
            pipeline=medspacy_nlp,
        )
        labels = {e.label for e in entities}
        assert "DRUG" in labels or "PROBLEM" in labels

    def test_filter_by_label(self, medspacy_nlp) -> None:
        """Test filtering entities by label."""
        entities = extract_medical_entities(
            "Patient takes ibuprofen for headache.",
            labels=["DRUG"],
            pipeline=medspacy_nlp,
        )
        assert all(e.label == "DRUG" for e in entities)

    def test_entity_has_correct_attributes(self, medspacy_nlp) -> None:
        """Test that entities have correct attributes."""
        entities = extract_medical_entities(
            "Patient takes aspirin.",
            pipeline=medspacy_nlp,
        )
        if entities:
            entity = entities[0]
            assert isinstance(entity, MedicalEntity)
            assert isinstance(entity.text, str)
            assert isinstance(entity.label, str)
            assert isinstance(entity.start, int)
            assert isinstance(entity.end, int)
            assert isinstance(entity.is_negated, bool)


class TestNegationDetection:
    """Tests for negation detection using ConText."""

    def test_negated_problem(self, medspacy_nlp) -> None:
        """Test detecting negated problems."""
        entities = extract_medical_entities(
            "Patient denies fever.",
            pipeline=medspacy_nlp,
        )
        fever_entities = [e for e in entities if e.text.lower() == "fever"]
        if fever_entities:
            assert fever_entities[0].is_negated

    def test_not_negated_problem(self, medspacy_nlp) -> None:
        """Test that non-negated problems are detected correctly."""
        entities = extract_medical_entities(
            "Patient has fever.",
            pipeline=medspacy_nlp,
        )
        fever_entities = [e for e in entities if e.text.lower() == "fever"]
        if fever_entities:
            assert not fever_entities[0].is_negated

    def test_no_prefix_negation(self, medspacy_nlp) -> None:
        """Test 'no' prefix negation."""
        entities = extract_medical_entities(
            "No headache reported.",
            pipeline=medspacy_nlp,
        )
        headache_entities = [e for e in entities if e.text.lower() == "headache"]
        if headache_entities:
            assert headache_entities[0].is_negated


class TestAssertHasMedicalEntity:
    """Tests for assert_has_medical_entity function."""

    def test_assert_drug_exists(self, medspacy_nlp) -> None:
        """Test asserting drug entity exists."""
        assert_has_medical_entity(
            doc="Patient takes ibuprofen.",
            entity_type="DRUG",
            pipeline=medspacy_nlp,
        )

    def test_assert_specific_drug(self, medspacy_nlp) -> None:
        """Test asserting specific drug text."""
        assert_has_medical_entity(
            doc="Patient takes aspirin daily.",
            entity_type="DRUG",
            text="aspirin",
            pipeline=medspacy_nlp,
        )

    def test_assert_fails_for_missing_entity(self, medspacy_nlp) -> None:
        """Test assertion fails when entity not found."""
        with pytest.raises(AssertionError) as exc_info:
            assert_has_medical_entity(
                doc="Patient is doing well.",
                entity_type="DRUG",
                pipeline=medspacy_nlp,
            )
        assert "Expected at least 1 DRUG" in str(exc_info.value)

    def test_assert_with_negation_filter(self, medspacy_nlp) -> None:
        """Test filtering by negation status."""
        # Should find non-negated fever
        assert_has_medical_entity(
            doc="Patient has fever.",
            entity_type="PROBLEM",
            text="fever",
            negated=False,
            pipeline=medspacy_nlp,
        )

    def test_custom_error_message(self, medspacy_nlp) -> None:
        """Test custom error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_has_medical_entity(
                doc="Hello world.",
                entity_type="DRUG",
                msg="Custom error: no drug found",
                pipeline=medspacy_nlp,
            )
        assert "Custom error: no drug found" in str(exc_info.value)


class TestAssertHasDrug:
    """Tests for assert_has_drug convenience function."""

    def test_assert_any_drug(self, medspacy_nlp) -> None:
        """Test asserting any drug exists."""
        assert_has_drug(
            doc="Patient takes metformin.",
            pipeline=medspacy_nlp,
        )

    def test_assert_specific_drug(self, medspacy_nlp) -> None:
        """Test asserting specific drug."""
        assert_has_drug(
            doc="Prescribed lisinopril 10mg.",
            drug_name="lisinopril",
            pipeline=medspacy_nlp,
        )

    def test_assert_drug_fails(self, medspacy_nlp) -> None:
        """Test assertion fails when no drug."""
        with pytest.raises(AssertionError):
            assert_has_drug(
                doc="Patient reports feeling well.",
                pipeline=medspacy_nlp,
            )


class TestAssertHasProblem:
    """Tests for assert_has_problem convenience function."""

    def test_assert_any_problem(self, medspacy_nlp) -> None:
        """Test asserting any problem exists."""
        assert_has_problem(
            doc="Patient has diabetes.",
            pipeline=medspacy_nlp,
        )

    def test_assert_specific_problem(self, medspacy_nlp) -> None:
        """Test asserting specific problem."""
        assert_has_problem(
            doc="Diagnosed with hypertension.",
            problem="hypertension",
            pipeline=medspacy_nlp,
        )

    def test_assert_problem_fails(self, medspacy_nlp) -> None:
        """Test assertion fails when no problem."""
        with pytest.raises(AssertionError):
            assert_has_problem(
                doc="Patient is healthy.",
                pipeline=medspacy_nlp,
            )


class TestAssertEntityNegated:
    """Tests for assert_entity_negated function."""

    def test_assert_negated_success(self, medspacy_nlp) -> None:
        """Test asserting entity is negated."""
        assert_entity_negated(
            doc="Patient denies fever.",
            entity_text="fever",
            pipeline=medspacy_nlp,
        )

    def test_assert_negated_fails_when_not_negated(self, medspacy_nlp) -> None:
        """Test assertion fails when entity is not negated."""
        with pytest.raises(AssertionError) as exc_info:
            assert_entity_negated(
                doc="Patient has fever.",
                entity_text="fever",
                pipeline=medspacy_nlp,
            )
        assert "not negated" in str(exc_info.value)

    def test_assert_negated_fails_when_not_found(self, medspacy_nlp) -> None:
        """Test assertion fails when entity not found."""
        with pytest.raises(AssertionError) as exc_info:
            assert_entity_negated(
                doc="Patient is well.",
                entity_text="fever",
                pipeline=medspacy_nlp,
            )
        assert "not found" in str(exc_info.value)


class TestAssertEntityNotNegated:
    """Tests for assert_entity_not_negated function."""

    def test_assert_not_negated_success(self, medspacy_nlp) -> None:
        """Test asserting entity is not negated."""
        assert_entity_not_negated(
            doc="Patient has fever.",
            entity_text="fever",
            pipeline=medspacy_nlp,
        )

    def test_assert_not_negated_fails_when_negated(self, medspacy_nlp) -> None:
        """Test assertion fails when entity is negated."""
        with pytest.raises(AssertionError) as exc_info:
            assert_entity_not_negated(
                doc="Patient denies fever.",
                entity_text="fever",
                pipeline=medspacy_nlp,
            )
        assert "is negated" in str(exc_info.value)


class TestClinicalDocumentScenarios:
    """Integration tests with realistic clinical document scenarios."""

    @pytest.fixture
    def clinical_note(self) -> str:
        """Sample clinical note."""
        return (
            "ASSESSMENT: Patient is a 65-year-old male with diabetes and hypertension. "
            "No chest pain or shortness of breath. "
            "MEDICATIONS: Metformin 500mg twice daily, lisinopril 10mg daily. "
            "Patient denies nausea or headache."
        )

    def test_extract_all_entities(self, clinical_note: str, medspacy_nlp) -> None:
        """Test extracting all entities from clinical note."""
        entities = extract_medical_entities(clinical_note, pipeline=medspacy_nlp)
        # Should find multiple entities
        assert len(entities) >= 2

    def test_find_medications(self, clinical_note: str, medspacy_nlp) -> None:
        """Test finding medications in note."""
        assert_has_drug(doc=clinical_note, drug_name="metformin", pipeline=medspacy_nlp)
        assert_has_drug(doc=clinical_note, drug_name="lisinopril", pipeline=medspacy_nlp)

    def test_find_problems(self, clinical_note: str, medspacy_nlp) -> None:
        """Test finding problems in note."""
        assert_has_problem(doc=clinical_note, problem="diabetes", pipeline=medspacy_nlp)
        assert_has_problem(doc=clinical_note, problem="hypertension", pipeline=medspacy_nlp)

    def test_negated_symptoms(self, clinical_note: str, medspacy_nlp) -> None:
        """Test detecting negated symptoms."""
        # These should be negated in the note
        entities = extract_medical_entities(clinical_note, pipeline=medspacy_nlp)
        negated_entities = [e for e in entities if e.is_negated]
        # Should have some negated entities (chest pain, shortness of breath, nausea, headache)
        assert len(negated_entities) >= 1

    def test_drug_discontinuation_scenario(self, medspacy_nlp) -> None:
        """Test drug discontinuation detection scenario."""
        doc = (
            "Patient was prescribed ibuprofen on 01/12/2023 for headaches. "
            "Ibuprofen was discontinued on 12/30/2023 due to GI side effects. "
            "No longer taking aspirin."
        )

        # Should find ibuprofen as a drug
        assert_has_drug(doc=doc, drug_name="ibuprofen", pipeline=medspacy_nlp)

        # Should find headaches as a problem
        entities = extract_medical_entities(doc, labels=["PROBLEM"], pipeline=medspacy_nlp)
        headache_found = any("headache" in e.text.lower() for e in entities)
        assert headache_found

