"""Three-stage pipeline: classify -> extract -> validate."""
import json
import pathlib
from datetime import datetime, timezone

from . import config
from .llm import get_provider
from .schema import empty_record, paginate, add_flag
from .validate import validate

PROMPTS = pathlib.Path(__file__).parent / "prompts"


def _load(name):
    return (PROMPTS / name).read_text()


def _parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(cleaned[start:end + 1])


def classify(text, provider):
    resp = provider.complete(_load("classify_v1.txt"), f"DOCUMENT TEXT:\n{text[:6000]}",
                             max_tokens=100)
    return _parse_json(resp.text), resp


def extract(text, provider):
    resp = provider.complete(_load("extract_v1.txt"), f"DOCUMENT TEXT:\n{text}")
    return _parse_json(resp.text), resp


def process(path, provider_name="mock", extract_model=None, classify_model=None, today=None):
    path = pathlib.Path(path)
    raw = path.read_text(errors="replace")
    paged, page_count = paginate(raw)

    record = empty_record(doc_id=path.stem, filename=path.name)
    record["document"]["page_count"] = page_count

    cls_provider = get_provider(provider_name, classify_model or config.CLASSIFY_MODEL)
    cls, cls_resp = classify(paged, cls_provider)
    record["document"]["doc_type"] = cls.get("doc_type", "unknown")
    record["document"]["doc_type_confidence"] = cls.get("doc_type_confidence", 0.0)

    total_in, total_out, total_ms = cls_resp.tokens_in, cls_resp.tokens_out, cls_resp.latency_ms
    ext_model = "n/a"

    # Early exit: don't spend extraction tokens on a document we don't handle.
    if record["document"]["doc_type"] == "out_of_scope":
        add_flag(record, "out_of_scope_document",
                 "Classified out of scope; extraction skipped", "info")
    else:
        ext_provider = get_provider(provider_name, extract_model or config.EXTRACT_MODEL)
        ext_model = ext_provider.model
        data, ext_resp = extract(paged, ext_provider)
        total_in += ext_resp.tokens_in
        total_out += ext_resp.tokens_out
        total_ms += ext_resp.latency_ms
        for key in ("parties", "dates", "deadlines", "obligations",
                    "monetary_amounts", "flags"):
            record[key] = data.get(key, []) or []
        if data.get("jurisdiction"):
            record["document"]["jurisdiction"].update(
                {k: v for k, v in data["jurisdiction"].items() if v})
        if data.get("document_date"):
            record["document"]["document_date"] = data["document_date"]

    record["extraction_meta"] = {
        "classify_model": cls_resp.model,
        "extract_model": ext_model,
        "prompt_version": config.PROMPT_VERSION,
        "tokens_in": total_in,
        "tokens_out": total_out,
        "cost_usd": (config.cost_usd(cls_resp.model, cls_resp.tokens_in, cls_resp.tokens_out)
                     + (config.cost_usd(ext_model, total_in - cls_resp.tokens_in,
                                        total_out - cls_resp.tokens_out)
                        if ext_model != "n/a" else 0)),
        "latency_ms": total_ms,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    return validate(record, raw, today=today)
