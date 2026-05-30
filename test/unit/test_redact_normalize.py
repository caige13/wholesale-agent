"""redact_normalize — front-door guardrail + unit normalization (spec §4, §10).

A pure regex function (no LLM): it strips *accidentally*-included phone/email
from the raw message before anything downstream sees it (recording only that the
guardrail fired, never the value), and normalizes spaced size units to their
attached form — while guaranteeing order quantities ("16oz", "3 cases", "1200")
survive untouched.
"""

import pytest

from src.domain.redaction import redact_normalize


def test_redacts_an_email_to_a_placeholder():
    result = redact_normalize("email me at bob@example.com please")
    assert "bob@example.com" not in result.clean_message
    assert "[REDACTED_EMAIL]" in result.clean_message


@pytest.mark.parametrize(
    "phone",
    ["555-123-4567", "(555) 123-4567", "555.123.4567", "+1 555 123 4567", "5551234567"],
)
def test_redacts_phone_numbers_in_common_formats(phone):
    result = redact_normalize(f"call {phone} when it arrives")
    assert phone not in result.clean_message
    assert "[REDACTED_PHONE]" in result.clean_message


def test_records_only_the_pii_type_not_the_value():
    result = redact_normalize("reach bob@example.com or 555-123-4567")
    assert "email" in result.pii_found
    assert "phone" in result.pii_found
    # never leak the raw value into state/traces
    assert "bob@example.com" not in result.pii_found
    assert "555-123-4567" not in result.pii_found


def test_normalizes_spaced_size_units_to_the_attached_form():
    assert redact_normalize("16 oz deli").clean_message == "16oz deli"
    assert redact_normalize("16 ounces deli").clean_message == "16oz deli"


def test_leaves_an_already_attached_quantity_intact():
    assert redact_normalize("16oz deli containers").clean_message == "16oz deli containers"


def test_does_not_treat_case_counts_as_size_units():
    assert redact_normalize("3 cases of salsa cups").clean_message == "3 cases of salsa cups"


def test_does_not_redact_order_quantities_as_phone_numbers():
    result = redact_normalize("I need 1200 16oz deli containers")
    assert "1200" in result.clean_message
    assert result.pii_found == []


def test_returns_text_unchanged_when_no_pii_or_units_present():
    result = redact_normalize("send me a case of limes")
    assert result.clean_message == "send me a case of limes"
    assert result.pii_found == []


def test_quantities_survive_redaction_when_pii_is_present():
    result = redact_normalize("deliver to 555-123-4567: 3 cases of 16oz deli")
    assert "3 cases of 16oz deli" in result.clean_message
    assert "phone" in result.pii_found