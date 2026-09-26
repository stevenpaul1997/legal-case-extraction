"""Streamlit demo. Paste or upload a document, see the validated case record."""
import json
import pathlib
import tempfile

import streamlit as st

from src import config
from src.pipeline import process

st.set_page_config(page_title="Legal Case Document Extraction", layout="wide")
st.title("Legal Case Document Extraction")
st.caption("Classify -> extract -> validate. Deadlines are computed in code, "
           "never inferred by the model.")

with st.sidebar:
    st.header("Run settings")
    provider = st.selectbox("Provider", ["groq", "mock"],
                            index=0 if config.GROQ_API_KEY else 1)
    model = st.selectbox("Extraction model",
                         ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    st.caption(f"Reasoning effort: {config.REASONING_EFFORT}")
    if provider == "groq" and not config.GROQ_API_KEY:
        st.warning("No GROQ_API_KEY found. Use the mock provider or set it in .env.")
    st.markdown("---")
    st.caption("Prompt version: " + config.PROMPT_VERSION)

SAMPLES = sorted((pathlib.Path(__file__).parent / "data" / "documents").glob("*.txt"))
sample = st.selectbox("Load a sample document",
                      ["(none)"] + [p.name for p in SAMPLES])
sample_text = ""
if sample != "(none)":
    sample_text = (pathlib.Path(__file__).parent / "data" / "documents" / sample).read_text()

uploaded = st.file_uploader("...or upload a .txt document", type=["txt"])
text = st.text_area("Document text", height=280,
                    value="" if uploaded else sample_text)

if st.button("Extract", type="primary"):
    content = uploaded.read().decode("utf-8", "replace") if uploaded else text
    if not content.strip():
        st.error("Give me a document first.")
        st.stop()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(content)
        tmp = fh.name
    with st.spinner("Extracting..."):
        record = process(tmp, provider_name=provider, extract_model=model)

    meta = record["extraction_meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document type", record["document"]["doc_type"])
    c2.metric("Latency", f"{meta['latency_ms']} ms")
    c3.metric("Cost", f"${meta['cost_usd']:.5f}")
    c4.metric("Tokens", meta["tokens_in"] + meta["tokens_out"])

    critical = [f for f in record["flags"] if f["severity"] == "critical"]
    if critical:
        st.error("Attorney review required")
        for f in critical:
            st.write(f"- **{f['type']}** — {f['detail']}")
    warnings = [f for f in record["flags"] if f["severity"] == "warning"]
    if warnings:
        with st.expander(f"{len(warnings)} warnings"):
            for f in warnings:
                st.write(f"- **{f['type']}** — {f['detail']}")

    st.subheader("Deadlines")
    if record["deadlines"]:
        st.dataframe([{
            "Obligation": d.get("obligation"),
            "Owed by": d.get("owed_by"),
            "Trigger": d.get("trigger_date"),
            "Rule": d.get("computation_rule") or "—",
            "Due": d.get("computed_due_date") or "NOT DETERMINABLE",
            "Basis": d.get("computation_basis"),
            "Review": "yes" if d.get("requires_attorney_review") else "no",
        } for d in record["deadlines"]], use_container_width=True)
    else:
        st.info("No deadlines found.")

    st.subheader("Parties")
    if record["parties"]:
        st.dataframe([{"Name": p.get("name"), "Role": p.get("role"),
                       "Confidence": p.get("confidence")} for p in record["parties"]],
                     use_container_width=True)

    with st.expander("Full case record (JSON)"):
        st.code(json.dumps(record, indent=2), language="json")
