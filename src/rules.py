"""Date arithmetic. Deterministic - the model never does this."""
import re
from datetime import date, timedelta, datetime

DAYS_PATTERNS = [
    (re.compile(r"(\d+)\s+business\s+days", re.I), "business"),
    (re.compile(r"(\d+)\s+court\s+days", re.I), "business"),
    (re.compile(r"(\d+)\s+(?:calendar\s+)?days", re.I), "calendar"),
]
MONTHS_PATTERN = re.compile(r"(\d+)\s+months?", re.I)


def parse_date(value):
    if isinstance(value, date):
        return value
    if not value or value in ("unknown", "null"):
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def add_business_days(start: date, n: int) -> date:
    d, added = start, 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def _days_in_month(year: int, month: int) -> int:
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]


def add_months(start: date, n: int) -> date:
    month = start.month - 1 + n
    year = start.year + month // 12
    month = month % 12 + 1
    return date(year, month, min(start.day, _days_in_month(year, month)))


def apply_rule(trigger_date, rule_text):
    """Return the due date implied by rule_text, or None if unparseable.

    None is a legitimate outcome: an unparseable rule sends the deadline to
    attorney review instead of producing a guessed date.
    """
    start = parse_date(trigger_date)
    if not start or not rule_text:
        return None
    for pattern, kind in DAYS_PATTERNS:
        m = pattern.search(rule_text)
        if m:
            n = int(m.group(1))
            return (add_business_days(start, n) if kind == "business"
                    else start + timedelta(days=n))
    m = MONTHS_PATTERN.search(rule_text)
    if m:
        return add_months(start, int(m.group(1)))
    return None


def is_weekend(value) -> bool:
    d = parse_date(value)
    return bool(d and d.weekday() >= 5)


def days_until(value, today=None):
    d = parse_date(value)
    if not d:
        return None
    return (d - (today or date.today())).days


MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}

_LONG_DATE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


_SQUASHED_LONG = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)(\d{1,2}),?(\d{4})", re.I)


def explicit_dates(text):
    """Every date written out in the document, as ISO strings.

    A deadline may only be triggered by a date a human can point to on the
    page. A trigger the document never states is a date the model derived,
    and deriving triggers is how invented deadlines get built.
    """
    found = set()
    for m in _LONG_DATE.finditer(text or ""):
        try:
            found.add(date(int(m.group(3)), MONTHS[m.group(1).lower()],
                           int(m.group(2))).isoformat())
        except ValueError:
            pass
    for m in _NUMERIC_DATE.finditer(text or ""):
        for month, day in ((int(m.group(1)), int(m.group(2))),
                           (int(m.group(2)), int(m.group(1)))):
            try:
                found.add(date(int(m.group(3)), month, day).isoformat())
            except ValueError:
                pass
    for m in _ISO_DATE.finditer(text or ""):
        found.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    # OCR damage spaces characters apart ("J u n e  2 2 , 2 0 2 6"), which hides
    # a date that is plainly on the page. Scan a whitespace-stripped copy too,
    # so degraded scans do not get their real dates rejected as untraceable.
    squashed = re.sub(r"\s+", "", text or "")
    for m in _SQUASHED_LONG.finditer(squashed):
        try:
            found.add(date(int(m.group(3)), MONTHS[m.group(1).lower()],
                           int(m.group(2))).isoformat())
        except (ValueError, KeyError):
            pass
    return found
