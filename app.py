"""Streamlit demo. Paste or upload a document, see the validated case record."""
import html
import json
import pathlib
import tempfile

import streamlit as st

from src import config
from src.pipeline import process

st.set_page_config(page_title="Legal Case Document Extraction",
                   page_icon="⚖️", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.4rem; max-width: 1180px;}
  #MainMenu, footer {visibility: hidden;}

  .lex-title {font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
              color: #15202b; margin-bottom: .2rem;}
  .lex-sub {color: #5a6675; font-size: .95rem; margin-bottom: .2rem;}
  .lex-rule {height: 3px; width: 56px; background: #1f3a5f; margin: .9rem 0 1.6rem;}

  .lex-section {font-size: .78rem; font-weight: 700; letter-spacing: .09em;
                text-transform: uppercase; color: #6b7785;
                margin: 2rem 0 .7rem;}

  .lex-metric {background: #f7f9fb; border: 1px solid #e3e8ee; border-radius: 8px;
               padding: .85rem 1rem;}
  .lex-metric .k {font-size: .7rem; letter-spacing: .08em; text-transform: uppercase;
                  color: #7a8694;}
  .lex-metric .v {font-size: 1.35rem; font-weight: 650; color: #15202b;
                  margin-top: .15rem;}

  .lex-card {border: 1px solid #e3e8ee; border-left: 4px solid #cbd3dc;
             border-radius: 8px; padding: .9rem 1.1rem; margin-bottom: .6rem;
             background: #fff;}
  .lex-card.ok {border-left-color: #2f7d5d;}
  .lex-card.warn {border-left-color: #c08a2e;}
  .lex-card.stop {border-left-color: #b4452f;}
  .lex-card .obl {font-weight: 650; color: #15202b; font-size: 1rem;}
  .lex-card .meta {color: #5a6675; font-size: .85rem; margin-top: .25rem;}

  .pill {display: inline-block; padding: .18rem .6rem; border-radius: 999px;
         font-size: .72rem; font-weight: 700; letter-spacing: .04em;}
  .pill.ok {background: #e6f2ec; color: #22684c;}
  .pill.warn {background: #fbf1dd; color: #8a6318;}
  .pill.stop {background: #fbe9e5; color: #8e3524;}

  .flag {border-radius: 6px; padding: .55rem .85rem; margin-bottom: .4rem;
         font-size: .88rem;}
  .flag.stop {background: #fdf1ee; border: 1px solid #f0cfc6; color: #7d3020;}
  .flag.warn {background: #fdf8ec; border: 1px solid #ecdcb8; color: #7a5a18;}
  .flag.info {background: #f4f6f9; border: 1px solid #e0e6ed; color: #4d5967;}
  .flag b {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           font-size: .8rem;}

  .party {display: flex; justify-content: space-between; align-items: center;
          border-bottom: 1px solid #edf0f4; padding: .55rem .2rem;}
  .party .nm {font-weight: 600; color: #15202b;}
  .party .rl {color: #6b7785; font-size: .85rem;}
  .conf {font-family: ui-monospace, Menlo, monospace; font-size: .8rem;
         color: #5a6675;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="lex-title">Legal Case Document Extraction</div>'
            '<div class="lex-sub">Classify → extract → validate. Deadlines are '
            'computed in code and never inferred by the model.</div>'
            '<div class="lex-rule"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="lex-section">Run settings</div>', unsafe_allow_html=True)
    provider = st.selectbox("Provider", ["groq", "mock"],
                            index=0 if config.GROQ_API_KEY else 1)
    model = st.selectbox("Extraction model",
                         ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    st.markdown(
        f'<div style="color:#6b7785;font-size:.82rem;line-height:1.7;margin-top:.8rem">'
        f'Reasoning effort · <b>{config.REASONING_EFFORT}</b><br>'
        f'Prompt version · <b>{config.PROMPT_VERSION}</b><br>'
        f'Review threshold · <b>{config.CONFIDENCE_REVIEW_THRESHOLD}</b></div>',
        unsafe_allow_html=True)
    if provider == "groq" and not config.GROQ_API_KEY:
        st.warning("No GROQ_API_KEY found. Use the mock provider or set it in .env.")

DOC_DIR = pathlib.Path(__file__).parent / "data" / "documents"
SAMPLES = sorted(DOC_DIR.glob("*.txt"))

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown('<div class="lex-section">Input</div>', unsafe_allow_html=True)
    sample = st.selectbox("Sample document",
                          ["(none)"] + [p.name for p in SAMPLES])
    sample_text = (DOC_DIR / sample).read_text() if sample != "(none)" else ""
    uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
    text = st.text_area("Document text", height=300,
                        value="" if uploaded else sample_text)
    run = st.button("Extract", type="primary")

if run:
    content = uploaded.read().decode("utf-8", "replace") if uploaded else text
    if not content.strip():
        st.error("Give me a document first.")
        st.stop()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(content)
        tmp = fh.name
    with st.spinner("Classifying, extracting, validating..."):
        record = process(tmp, provider_name=provider, extract_model=model)
    st.session_state["record"] = record

record = st.session_state.get("record")

with right:
    if not record:
        st.markdown('<div class="lex-section">Case record</div>'
                    '<div style="color:#8995a3;border:1px dashed #d8dee6;'
                    'border-radius:8px;padding:2.4rem;text-align:center">'
                    'Load a document and press Extract.</div>',
                    unsafe_allow_html=True)
    else:
        meta = record["extraction_meta"]
        st.markdown('<div class="lex-section">Run</div>', unsafe_allow_html=True)
        cells = [("Document type", record["document"]["doc_type"].replace("_", " ")),
                 ("Latency", f"{meta['latency_ms'] / 1000:.1f}s"),
                 ("Cost", f"${meta['cost_usd']:.5f}"),
                 ("Tokens", f"{meta['tokens_in'] + meta['tokens_out']:,}")]
        for col, (k, v) in zip(st.columns(4), cells):
            col.markdown(f'<div class="lex-metric"><div class="k">{html.escape(k)}</div>'
                         f'<div class="v">{html.escape(str(v))}</div></div>',
                         unsafe_allow_html=True)

        sev_class = {"critical": "stop", "warning": "warn", "info": "info"}
        flags = record.get("flags", [])
        if flags:
            crit = sum(1 for f in flags if f["severity"] == "critical")
            st.markdown(f'<div class="lex-section">Review queue · {len(flags)} flag'
                        f'{"s" if len(flags) != 1 else ""}'
                        f'{f" · {crit} critical" if crit else ""}</div>',
                        unsafe_allow_html=True)
            order = {"critical": 0, "warning": 1, "info": 2}
            for f in sorted(flags, key=lambda x: order.get(x["severity"], 3)):
                st.markdown(
                    f'<div class="flag {sev_class.get(f["severity"], "info")}">'
                    f'<b>{html.escape(f["type"])}</b> — {html.escape(f["detail"])}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="lex-section">Deadlines</div>', unsafe_allow_html=True)
        if not record["deadlines"]:
            st.markdown('<div style="color:#8995a3">None found in this document.</div>',
                        unsafe_allow_html=True)
        for d in record["deadlines"]:
            due = d.get("computed_due_date")
            if due and not d.get("requires_attorney_review"):
                tone, pill = "ok", f'<span class="pill ok">DUE {due}</span>'
            elif due:
                tone, pill = "warn", f'<span class="pill warn">DUE {due} · REVIEW</span>'
            else:
                tone, pill = "stop", '<span class="pill stop">NOT DETERMINABLE</span>'
            rule = d.get("computation_rule") or "no period stated in document"
            st.markdown(
                f'<div class="lex-card {tone}">'
                f'<div class="obl">{html.escape(str(d.get("obligation", "")))}</div>'
                f'<div class="meta">Owed by {html.escape(str(d.get("owed_by", "—")))} · '
                f'triggered {html.escape(str(d.get("trigger_date", "—")))} · '
                f'{html.escape(str(rule))}<br>'
                f'basis: {html.escape(str(d.get("computation_basis", "—")))}</div>'
                f'<div style="margin-top:.55rem">{pill}</div></div>',
                unsafe_allow_html=True)

        if record["parties"]:
            st.markdown('<div class="lex-section">Parties</div>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="party"><div><div class="nm">'
                f'{html.escape(str(p.get("name", "")))}</div>'
                f'<div class="rl">{html.escape(str(p.get("role", "")).replace("_", " "))}'
                f'</div></div><div class="conf">{p.get("confidence", 0):.2f}</div></div>'
                for p in record["parties"])
            st.markdown(rows, unsafe_allow_html=True)

        if record.get("monetary_amounts"):
            st.markdown('<div class="lex-section">Amounts</div>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="party"><div class="nm">'
                f'{html.escape(str(a.get("label", "")))}</div>'
                f'<div class="conf">${a.get("amount", 0):,.2f}</div></div>'
                for a in record["monetary_amounts"])
            st.markdown(rows, unsafe_allow_html=True)

        with st.expander("Full case record (JSON)"):
            st.code(json.dumps(record, indent=2), language="json")
