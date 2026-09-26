"""PII detection. Regex, not trust."""
import re

PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "BANK_ACCOUNT": re.compile(r"(?i)\b(?:account|acct)\.?\s*(?:no\.?|number|#)?\s*:?\s*(\d{8,17})\b"),
    "DL": re.compile(r"(?i)\b(?:driver'?s?\s+licen[cs]e|dl)\s*(?:no\.?|number|#)?\s*:?\s*([A-Z0-9]{6,12})\b"),
    "DOB": re.compile(r"(?i)\b(?:date of birth|d\.?o\.?b\.?)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
}


def find_pii(text: str):
    """Yield (kind, matched_value) for every PII hit in text."""
    for kind, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text or ""):
            yield kind, (m.group(1) if m.groups() else m.group(0))


def redact_text(text: str) -> str:
    out = text or ""
    for kind, pattern in PII_PATTERNS.items():
        if pattern.groups:
            out = pattern.sub(
                lambda m, k=kind: m.group(0).replace(m.group(1), f"[REDACTED:{k}]"), out)
        else:
            out = pattern.sub(f"[REDACTED:{kind}]", out)
    return out
