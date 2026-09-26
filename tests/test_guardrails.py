"""The guardrails are the product. These tests are what keep them honest."""
import pathlib
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.rules import apply_rule, add_business_days, is_weekend  # noqa: E402
from src.validate import validate  # noqa: E402
from src.schema import empty_record  # noqa: E402


def _record_with_deadline(**kw):
    rec = empty_record("t", "t.txt")
    rec["document"]["doc_type"] = "summons"
    rec["document"]["jurisdiction"]["case_number"] = "CV-1"
    rec["parties"] = [{"name": "A Party", "role": "defendant",
                       "source": {"page": 1, "quote": "A Party"}, "confidence": 0.9}]
    base = {"obligation": "file response", "owed_by": "Defendant",
            "trigger_event": "service", "trigger_date": "2026-03-12",
            "computation_rule": None, "computed_due_date": None,
            "computation_basis": "not_determinable", "requires_attorney_review": True,
            "source": {"page": 1, "quote": "Served on Defendant"}, "confidence": 0.9}
    base.update(kw)
    rec["deadlines"] = [base]
    return rec


SRC = "Served on Defendant: March 12, 2026. A Party appears."


def test_invented_deadline_is_stripped():
    """The core safety property: no date without a stated rule."""
    rec = validate(_record_with_deadline(computed_due_date="2026-04-11"), SRC,
                   today=date(2026, 3, 13))
    assert rec["deadlines"][0]["computed_due_date"] is None
    assert rec["deadlines"][0]["requires_attorney_review"] is True
    assert any(f["type"] == "undeterminable_deadline" for f in rec["flags"])


def test_code_overrides_model_arithmetic():
    rec = validate(_record_with_deadline(
        computation_basis="stated_in_document",
        computation_rule="within 30 calendar days after service",
        computed_due_date="2026-04-25"), SRC, today=date(2026, 3, 13))
    assert rec["deadlines"][0]["computed_due_date"] == "2026-04-11"
    assert any(f["type"] == "ambiguous_date" for f in rec["flags"])


def test_unquotable_value_loses_confidence():
    rec = empty_record("t", "t.txt")
    rec["document"]["doc_type"] = "summons"
    rec["parties"] = [{"name": "Ghost Party", "role": "defendant",
                       "source": {"page": 1, "quote": "never appears in source"},
                       "confidence": 0.95}]
    rec = validate(rec, SRC)
    assert rec["parties"][0]["confidence"] == 0.0
    assert any(f["type"] == "unsupported_extraction" for f in rec["flags"])


def test_pii_is_redacted_from_output():
    rec = empty_record("t", "t.txt")
    rec["document"]["doc_type"] = "correspondence"
    rec["parties"] = [{"name": "Client SSN 412-55-9087", "role": "third_party",
                       "source": {"page": 1, "quote": "Client SSN 412-55-9087"},
                       "confidence": 0.9}]
    rec = validate(rec, "Client SSN 412-55-9087")
    assert "412-55-9087" not in str(rec)
    assert any(f["type"] == "pii_detected" for f in rec["flags"])


def test_weekend_landing_goes_to_review():
    rec = validate(_record_with_deadline(
        computation_basis="stated_in_document",
        computation_rule="within 30 calendar days after service",
        computed_due_date="2026-04-11"), SRC, today=date(2026, 3, 13))
    assert is_weekend("2026-04-11")
    assert rec["deadlines"][0]["requires_attorney_review"] is True


def test_unparseable_rule_does_not_produce_a_date():
    rec = validate(_record_with_deadline(
        computation_basis="stated_in_document",
        computation_rule="promptly and without undue delay",
        computed_due_date="2026-04-01"), SRC, today=date(2026, 3, 13))
    assert rec["deadlines"][0]["computed_due_date"] is None


def test_rule_parsing():
    assert apply_rule("2026-03-12", "within 30 calendar days").isoformat() == "2026-04-11"
    assert apply_rule("2026-05-01", "within 10 business days") == add_business_days(
        date(2026, 5, 1), 10)
    assert apply_rule("2026-01-31", "within 1 month").isoformat() == "2026-02-28"
    assert apply_rule("2026-03-12", "as soon as practicable") is None
    assert apply_rule("unknown", "within 30 days") is None


def test_chained_trigger_is_rejected():
    """Document 4's failure: the model used a date it had computed itself as the
    trigger for a second deadline. Real fragments, invented result."""
    src = ("This offer expires 14 days from the date of this letter.\n"
           "April 2, 2026")
    rec = empty_record("t", "t.txt")
    rec["document"]["doc_type"] = "demand_letter"
    rec["parties"] = [{"name": "A Party", "role": "insurer",
                       "source": {"page": 1, "quote": "April 2, 2026"}, "confidence": 0.9}]
    rec["deadlines"] = [{"obligation": "file lawsuit after offer expires",
                         "owed_by": "Plaintiff", "trigger_event": "expiry",
                         "trigger_date": "2026-04-16",   # never appears in the document
                         "computation_rule": "14 days",
                         "computed_due_date": "2026-04-30",
                         "computation_basis": "stated_in_document",
                         "requires_attorney_review": False,
                         "source": {"page": 1, "quote": "expires 14 days"},
                         "confidence": 0.9}]
    rec = validate(rec, src, today=date(2026, 4, 3))
    assert rec["deadlines"][0]["computed_due_date"] is None
    assert rec["deadlines"][0]["computation_basis"] == "not_determinable"
    assert any(f["type"] == "untraceable_trigger" for f in rec["flags"])


def test_traceable_trigger_still_computes():
    src = "This offer expires 14 days from the date of this letter.\nApril 2, 2026"
    rec = empty_record("t", "t.txt")
    rec["document"]["doc_type"] = "demand_letter"
    rec["parties"] = [{"name": "A Party", "role": "insurer",
                       "source": {"page": 1, "quote": "April 2, 2026"}, "confidence": 0.9}]
    rec["deadlines"] = [{"obligation": "offer expires", "owed_by": "Claimant",
                         "trigger_event": "letter date", "trigger_date": "2026-04-02",
                         "computation_rule": "14 days from the date of this letter",
                         "computed_due_date": "2026-04-16",
                         "computation_basis": "stated_in_document",
                         "requires_attorney_review": False,
                         "source": {"page": 1, "quote": "expires 14 days"},
                         "confidence": 0.9}]
    rec = validate(rec, src, today=date(2026, 4, 3))
    assert rec["deadlines"][0]["computed_due_date"] == "2026-04-16"
