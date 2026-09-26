"""LLM providers. Groq for real runs, Mock so the pipeline runs offline."""
import json
import re
import time
from dataclasses import dataclass

import requests

from . import config


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class GroqProvider:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _api_message(response):
        try:
            body = response.json()
            return (body.get("error", {}).get("message")
                    or body.get("error", {}).get("code") or str(body)[:400])
        except Exception:
            return response.text[:400]

    MAX_RETRIES = 5

    def complete(self, system, user, max_tokens=None, temperature=0.0):
        max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env or use --provider mock.")
        started = time.time()
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": temperature,
                   "max_tokens": max_tokens,
                   "response_format": {"type": "json_object"}}
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = config.REASONING_EFFORT

        for attempt in range(self.MAX_RETRIES):
            r = requests.post(
                config.GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload, timeout=120)

            if r.status_code == 429:
                # Rate limited. Honour Retry-After when present, else back off.
                wait = float(r.headers.get("retry-after", 0) or 0) or (2 ** attempt) * 2
                if attempt == self.MAX_RETRIES - 1:
                    raise RuntimeError(f"Rate limited after {self.MAX_RETRIES} attempts. "
                                       f"Last message: {self._api_message(r)}")
                print(f"      rate limited, waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(wait)
                continue

            if r.status_code >= 400:
                message = self._api_message(r)
                # Strict JSON mode fails when the model runs out of budget
                # mid-object. Retry once in free-text mode and parse leniently -
                # the downstream parser already tolerates stray prose.
                if ("JSON" in message and payload.get("response_format")):
                    print(f"      strict JSON mode failed, retrying without it")
                    payload.pop("response_format")
                    payload["messages"][0]["content"] += (
                        "\n\nIMPORTANT: respond with the JSON object only. "
                        "No preamble, no explanation, no markdown fences.")
                    continue
                # Surface what the API actually said - the status code alone
                # is not a diagnosis.
                raise RuntimeError(f"HTTP {r.status_code} from Groq: {message}")
            break

        data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=int((time.time() - started) * 1000),
        )


class MockProvider:
    """Crude rule-based stand-in so the pipeline and validators are testable
    without an API key. Deliberately imperfect - it makes some of the same
    mistakes a real model makes, which is useful for exercising guardrails."""

    def __init__(self, model="mock"):
        self.model = model

    def complete(self, system, user, max_tokens=4000, temperature=0.0):
        started = time.time()
        body = user.split("DOCUMENT TEXT:", 1)[-1]
        payload = (self._classify(body) if "Return only the document type"
                   in system else self._extract(body))
        return LLMResponse(
            text=json.dumps(payload),
            model=self.model,
            tokens_in=len(user) // 4,
            tokens_out=len(json.dumps(payload)) // 4,
            latency_ms=int((time.time() - started) * 1000) + 1,
        )

    @staticmethod
    def _classify(text):
        t = text.lower()
        for kw, dt in [("summons", "summons"), ("complaint", "complaint"),
                       ("settlement agreement", "settlement_agreement"),
                       ("demand", "demand_letter"),
                       ("request for production", "discovery_request"),
                       ("interrogator", "discovery_request"),
                       ("it is hereby ordered", "court_order"),
                       ("order", "court_order")]:
            if kw in t:
                return {"doc_type": dt, "doc_type_confidence": 0.8}
        if "statement of account" in t or "amount due" in t or "subscribe" in t:
            return {"doc_type": "out_of_scope", "doc_type_confidence": 0.7}
        return {"doc_type": "correspondence", "doc_type_confidence": 0.5}

    @staticmethod
    def _extract(text):
        rec = {"parties": [], "dates": [], "deadlines": [],
               "obligations": [], "monetary_amounts": [], "flags": [],
               "jurisdiction": {}}
        m = re.search(r"Case No\.?\s*([A-Z0-9\-]+)", text)
        if m:
            rec["jurisdiction"]["case_number"] = m.group(1)
        for name, role in re.findall(r"^([A-Z][A-Z\s,\.'\-]{3,50}),\s*\n?\s*(Plaintiff|Defendant)",
                                     text, re.M):
            rec["parties"].append({
                "name": name.strip().rstrip(","),
                "role": role.lower(),
                "entity_type": "organization" if re.search(r"\b(LLC|INC|CORP)\b", name) else "individual",
                "represented_by": None,
                "source": {"page": 1, "quote": name.strip().rstrip(",")},
                "confidence": 0.75,
            })
        m = re.search(r"Served on Defendant:\s*([A-Za-z]+ \d{1,2}, \d{4})", text)
        if m:
            from datetime import datetime as _dt
            iso = _dt.strptime(m.group(1), "%B %d, %Y").date().isoformat()
            rule = None
            rm = re.search(r"within (\d+ (?:calendar |business |court )?days)", text, re.I)
            if rm:
                rule = f"within {rm.group(1)} after service"
            # Deliberate flaw: the mock guesses 30 days when no rule is stated.
            rec["deadlines"].append({
                "obligation": "file written response",
                "owed_by": "Defendant",
                "trigger_event": "service",
                "trigger_date": iso,
                "computation_rule": rule,
                "computed_due_date": None if rule else "GUESS",
                "computation_basis": "stated_in_document" if rule else "not_determinable",
                "requires_attorney_review": True,
                "source": {"page": 1, "quote": m.group(0)},
                "confidence": 0.8,
            })
            if rec["deadlines"][-1]["computed_due_date"] == "GUESS":
                from datetime import timedelta as _td
                from .rules import parse_date
                rec["deadlines"][-1]["computed_due_date"] = (
                    parse_date(iso) + _td(days=30)).isoformat()
        for label, amt in re.findall(r"(?i)(damages|settlement sum|total|amount)[^\$]{0,30}\$([\d,]+(?:\.\d{2})?)", text):
            rec["monetary_amounts"].append({
                "label": label.lower(), "amount": float(amt.replace(",", "")),
                "currency": "USD", "source": {"page": 1, "quote": f"${amt}"},
                "confidence": 0.7})
        return rec


def get_provider(name, model=None):
    if name == "mock":
        return MockProvider()
    return GroqProvider(model or config.EXTRACT_MODEL)
