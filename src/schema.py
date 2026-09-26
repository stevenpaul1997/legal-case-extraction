"""Case record shape and helpers for walking it."""

DOC_TYPES = [
    "complaint", "summons", "demand_letter", "settlement_agreement",
    "discovery_request", "court_order", "correspondence",
    "out_of_scope", "unknown",
]

FLAG_TYPES = [
    "missing_critical_field", "ambiguous_date", "conflicting_dates",
    "deadline_within_7_days", "deadline_passed", "undeterminable_deadline",
    "pii_detected",
    "degraded_source_quality", "out_of_scope_document", "unsupported_extraction",
    "quote_not_verbatim", "untraceable_trigger",
]

VALUE_ARRAYS = ["parties", "dates", "deadlines", "obligations", "monetary_amounts"]


def empty_record(doc_id="", filename=""):
    return {
        "document": {
            "doc_id": doc_id,
            "source_filename": filename,
            "doc_type": "unknown",
            "doc_type_confidence": 0.0,
            "jurisdiction": {"state": "unknown", "court": "unknown",
                             "county": "unknown", "case_number": "unknown"},
            "document_date": "unknown",
            "page_count": 0,
        },
        "parties": [], "dates": [], "deadlines": [],
        "obligations": [], "monetary_amounts": [], "flags": [],
        "extraction_meta": {},
    }


def iter_values(record):
    """Every extracted item that should carry a source + confidence."""
    for key in VALUE_ARRAYS:
        for item in record.get(key, []) or []:
            yield key, item


def add_flag(record, flag_type, detail, severity="warning"):
    record.setdefault("flags", []).append(
        {"type": flag_type, "detail": detail, "severity": severity})


def mark_for_review(record, item):
    item["requires_attorney_review"] = True


def paginate(text, page_size_chars=3000):
    """Insert [PAGE n] markers so the model can cite a page."""
    if "[PAGE " in text:
        return text, text.count("[PAGE ")
    chunks = [text[i:i + page_size_chars] for i in range(0, max(len(text), 1), page_size_chars)]
    out = "\n".join(f"[PAGE {i+1}]\n{c}" for i, c in enumerate(chunks))
    return out, len(chunks)
