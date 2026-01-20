"""Pytest configuration for pytest-nlp tests."""

import pytest


# Sample documents for testing
@pytest.fixture
def drug_document() -> str:
    """Sample medical document about drug prescription and discontinuation."""
    return (
        "Patient was prescribed Ibuprofen on 01/12/2023 for headaches. "
        "Patient reported mild side effects. "
        "Ibuprofen was discontinued on 12/30/2023."
    )


@pytest.fixture
def simple_sentence() -> str:
    """Simple sentence for basic testing."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def dosage_document() -> str:
    """Medical document with dosage information."""
    return (
        "Patient was admitted with severe pain. "
        "Prescribed Ibuprofen 200mg twice daily. "
        "After one week, dosage increased to 400mg. "
        "Patient showed improvement and was discharged."
    )

