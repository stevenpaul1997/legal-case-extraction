"""Stage 3: deterministic guardrails.

A guardrail written as a prompt instruction is a request.
A guardrail written in code is a constraint. This module is the constraints.
"""
import json
import re

from . import config
from .pii import PII_PATTERNS, redact_text
from .rules import apply_rule, is_weekend, days_until, parse_date, explicit_dates
from .schema import add_flag, iter_values, mark_for_review


def _normalize(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _words(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _word_coverage(quote, source_text):
    """Share of the quote's words that appear anywhere in the source.

    1.0 means every word is present (the model reordered or stitched).
    A fabricated quote scores low because its invented words are absent.
    """
    qw = _words(quote)
    if not qw:
        return 0.0
    sw = set(_words(source_text))
    return sum(1 for w in qw if w in sw) / len(qw)


OCR_SIGNATURE = re.compile(r"[A-Za-z][01][A-Za-z]")


def _ocr_damage_ratio(text):
    """Fraction of words showing digit-for-letter substitution (0 for O, 1 for I)."""
    words = re.findall(r"[A-Za-z0-9]{3,}", text or "")
    if not words:
        return 0.0
    damaged = sum(1 for w in words if OCR_SIGNATURE.search(w))
    return damaged / len(words)


def validate(record, source_text, today=None):
    """Amend the record in place and append flags. Returns the record."""
    norm_source = _normalize(source_text)

    # 1. Provenance: every quote must trace back to the source.
    #    Exact match is the strong signal. Models legitimately stitch across
    #    lines (dropping text that sits between two fragments), so a quote whose
    #    words are all present but not contiguous is downgraded and flagged
    #    rather than discarded - discarding it punished correct extractions.
    for array_name, item in iter_values(record):
        quote = (item.get("source") or {}).get("quote", "")
        if not quote:
            item["confidence"] = 0.0
            add_flag(record, "unsupported_extraction",
                     f"{array_name}: value extracted with no source quote", "critical")
        elif _normalize(quote) in norm_source:
            continue
        elif _word_coverage(quote, source_text) >= 0.9:
            item["confidence"] = min(item.get("confidence", 1.0), 0.8)
            add_flag(record, "quote_not_verbatim",
                     f"{array_name}: quote stitched from non-contiguous text -> "
                     f"{quote[:60]}", "warning")
        else:
            item["confidence"] = 0.0
            add_flag(record, "unsupported_extraction",
                     f"{array_name}: quote not found in source -> {quote[:60]}", "critical")

    # 2. Deadline discipline: strip any date the document did not justify.
    for d in record.get("deadlines", []):
        if d.get("computation_basis") == "not_determinable" and d.get("computed_due_date"):
            add_flag(record, "undeterminable_deadline",
                     f"Model computed a date with no stated rule: {d.get('obligation')} "
                     f"({d['computed_due_date']}) - removed", "critical")
            d["computed_due_date"] = None
            d["requires_attorney_review"] = True
        elif d.get("computation_basis") == "not_determinable":
            d["requires_attorney_review"] = True
            add_flag(record, "undeterminable_deadline",
                     f"No stated period for: {d.get('obligation')}", "warning")

    # 2b. Trigger traceability. A stated-basis deadline must hang off a date
    #     the document actually contains. When the model uses a date it derived
    #     itself as the trigger, it is chaining one computation onto another -
    #     which is how an invented deadline gets built out of real fragments.
    doc_dates = explicit_dates(source_text)
    for d in record.get("deadlines", []):
        if d.get("computation_basis") not in ("stated_in_document",
                                              "rule_cited_in_document"):
            continue
        trigger = d.get("trigger_date")
        if trigger and trigger != "unknown" and trigger not in doc_dates:
            add_flag(record, "untraceable_trigger",
                     f"Trigger date {trigger} for '{d.get('obligation')}' does not "
                     f"appear in the document; computed date removed", "critical")
            d["computed_due_date"] = None
            d["computation_basis"] = "not_determinable"
            d["requires_attorney_review"] = True

    # 3. Recompute stated-basis deadlines in code. The model reads the rule,
    #    Python does the arithmetic.
    for d in record.get("deadlines", []):
        if d.get("computation_basis") in ("stated_in_document", "rule_cited_in_document"):
            expected = apply_rule(d.get("trigger_date"), d.get("computation_rule"))
            if expected is None:
                d["computed_due_date"] = None
                d["requires_attorney_review"] = True
                add_flag(record, "undeterminable_deadline",
                         f"Rule not machine-parseable: {d.get('computation_rule')!r}", "warning")
            elif expected.isoformat() != d.get("computed_due_date"):
                add_flag(record, "ambiguous_date",
                         f"Model said {d.get('computed_due_date')}, code computed "
                         f"{expected.isoformat()} for {d.get('obligation')} - code value used",
                         "warning")
                d["computed_due_date"] = expected.isoformat()

    # 4. Weekend landing is a court-rule question, not ours.
    for d in record.get("deadlines", []):
        if d.get("computed_due_date") and is_weekend(d["computed_due_date"]):
            d["requires_attorney_review"] = True
            add_flag(record, "ambiguous_date",
                     f"Computed date {d['computed_due_date']} falls on a weekend; "
                     "roll-forward is a court rule", "warning")

    # 5. PII sweep. Scan both the source (so the flag fires even if the model
    #    dropped the value) and the emitted record (so nothing leaks through).
    blob = json.dumps(record)
    hits = {kind for kind, pattern in PII_PATTERNS.items()
            if pattern.search(blob) or pattern.search(source_text or "")}
    if hits:
        redacted = json.loads(redact_text(blob))
        record.clear()
        record.update(redacted)
        for kind in sorted(hits):
            add_flag(record, "pii_detected", f"{kind} redacted from output", "critical")

    # 5b. Source quality. OCR damage is detectable in code: letter-shaped
    #     digits inside words are the usual signature.
    if _ocr_damage_ratio(source_text) > 0.02:
        add_flag(record, "degraded_source_quality",
                 "Source shows OCR-style character substitution; "
                 "extraction confidence reduced", "warning")
        for _, item in iter_values(record):
            item["confidence"] = min(item.get("confidence", 1.0), 0.6)

    # 6. Urgency surfacing.
    for d in record.get("deadlines", []):
        n = days_until(d.get("computed_due_date"), today)
        if n is None:
            continue
        if n < 0:
            add_flag(record, "deadline_passed",
                     f"{d.get('obligation')} was due {d['computed_due_date']} "
                     f"({abs(n)} days ago)", "critical")
        elif n <= config.URGENT_DEADLINE_DAYS:
            add_flag(record, "deadline_within_7_days",
                     f"{d.get('obligation')} due {d['computed_due_date']} (in {n} days)",
                     "critical")

    # 7. Confidence gate.
    for _, item in iter_values(record):
        if item.get("confidence", 0) < config.CONFIDENCE_REVIEW_THRESHOLD:
            mark_for_review(record, item)

    # 8. Critical-field presence.
    juris = record.get("document", {}).get("jurisdiction", {})
    if juris.get("case_number") in (None, "", "unknown") and \
            record.get("document", {}).get("doc_type") not in ("out_of_scope", "demand_letter",
                                                               "correspondence"):
        add_flag(record, "missing_critical_field", "case_number not found", "warning")
    if not record.get("parties") and record.get("document", {}).get("doc_type") != "out_of_scope":
        add_flag(record, "missing_critical_field", "no parties extracted", "critical")

    # De-duplicate flags.
    seen, unique = set(), []
    for f in record.get("flags", []):
        key = (f.get("type"), f.get("detail"))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    record["flags"] = unique
    return record
