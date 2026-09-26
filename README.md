# Legal Case Document Extraction

Takes a legal document (complaint, summons, demand letter, settlement agreement,
court order, discovery request) and returns a validated structured case record:
parties, dates, deadlines, obligations, monetary amounts, and review flags.

Built as a working prototype with an eval harness, not a demo.

## The design decision that matters

**The model reads the rule. Python does the date arithmetic. Neither guesses.**

If a document says "within 30 calendar days after service", the model extracts
that rule and the trigger date, and `src/rules.py` computes the due date. If the
document does *not* state a period, the system returns `not_determinable` and
flags it for attorney review, even though the model almost certainly "knows" a
plausible answer. A confidently wrong filing deadline is the worst output this
product can produce, so the architecture refuses to produce one.

That refusal is enforced in code (`src/validate.py`), not in the prompt. A
guardrail written as a prompt instruction is a request. A guardrail written in
code is a constraint.

## Architecture

    Stage 1  classify   cheap model, doc type only, early exit if out of scope
    Stage 2  extract    schema-constrained extraction with mandatory provenance
    Stage 3  validate   deterministic guardrails, no model involved

Stage 3 enforces:

1. **Provenance** — every value must quote the source; unquotable values are
   zeroed out and flagged as `unsupported_extraction`
2. **Deadline discipline** — a computed date with no stated basis is deleted
3. **Recomputation** — stated-basis deadlines are recomputed in Python; the
   model's arithmetic never survives
4. **Weekend landings** — flagged, never silently rolled forward (that's a court
   rule, not a document rule)
5. **PII** — regex sweep over source and output, values redacted
6. **Urgency** — deadlines inside 7 days, and already-passed deadlines, raised
   as critical
7. **Confidence gate** — anything under 0.70 goes to attorney review
8. **Critical fields** — missing case number or parties flagged

## Quick start

    pip install -r requirements.txt
    cp .env.example .env        # add your GROQ_API_KEY

    # single document
    python run.py data/documents/03_summons_no_rule.txt --provider groq

    # full eval sweep
    python -m eval.run_eval --provider groq --extract-model llama-3.3-70b-versatile

    # UI
    streamlit run app.py

    # guardrail tests
    python -m pytest tests/ -q

Runs without an API key using `--provider mock`, a deliberately imperfect
rule-based stand-in that makes some of the mistakes a real model makes. Useful
for exercising the validators and for showing that the harness catches failures.

## Evaluation

12 test documents in `data/documents/`, each with ground truth in
`data/ground_truth/`. The set is deliberately trap-heavy:

| Document | Tests |
|---|---|
| 01 complaint (clean) | baseline |
| 02 summons, rule stated | correct computation (lands on a Saturday) |
| 03 summons, **no rule** | refusal to guess — the key test |
| 04 demand letter | computation anchored to letterhead date |
| 05 settlement | staggered deadlines + one conditional with no trigger |
| 06 correspondence | conflicting letterhead vs signature dates |
| 07 discovery (OCR-garbled) | graceful degradation, confidence drop |
| 08 medical bill | scope control, early exit |
| 09 amended complaint | 3 plaintiffs, 2 defendants, a DBA, 3 counsel, a judge |
| 10 intake letter | SSN, DOB, DL, bank account redaction |
| 11 court order | deadline landing on a weekend |
| 12 case management order | "15th day of the month following entry" |

### Metrics

`results/eval_results.csv` gets one row per document per model per prompt
version.

- **Unsafe deadline count** — the headline number. Counts wrong dates, dates
  produced without a stated basis, and missed deadlines. Target: zero.
- **Provenance validity %** — share of extracted values whose quote genuinely
  appears in the source. Automatable fabrication check.
- **Flag recall** — share of expected flags raised.
- **Party / amount precision and recall**
- **Cost and latency per document**

Change one thing at a time (prompt version or model), rerun, compare rows.

## Layout

    src/pipeline.py     three stages wired together
    src/validate.py     the guardrails
    src/rules.py        date arithmetic
    src/pii.py          PII patterns
    src/llm.py          Groq + mock providers
    src/prompts/        versioned prompts (extract_v1, classify_v1)
    eval/run_eval.py    sweep + summary
    eval/score.py       scoring against ground truth
    tests/              guardrail unit tests

## Known limits

- Text input only; OCR is simulated in the test set rather than performed
- `conflicting_dates` detection is prompt-side only, not enforced in code
- Jurisdiction-specific procedural rules are deliberately out of scope
- Cost figures in `src/config.py` need verifying against current pricing
