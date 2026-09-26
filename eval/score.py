"""Score one extracted record against ground truth."""
import re


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


STOPWORDS = {"the", "a", "an", "of", "to", "and", "or", "with", "upon", "on",
             "in", "for", "by", "this", "that", "shall", "must", "is", "be"}


def _tokens(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in STOPWORDS}


def _similarity(a, b):
    """Containment score: how much of the smaller label is inside the larger.

    Labels are model-written, so 'produce requested documents' and
    'produce documents' are the same obligation. Exact-string matching
    treated them as two different ones and scored a correct extraction
    as both a miss and an unexplained date.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _prf(predicted, expected):
    tp = len(predicted & expected)
    p = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    r = tp / len(expected) if expected else 1.0
    return round(p, 3), round(r, 3)


def score(record, gt, source_text):
    res = {"doc_id": gt["doc_id"]}

    res["doc_type_correct"] = record["document"]["doc_type"] == gt["doc_type"]

    pred_parties = {_norm(p.get("name")) for p in record.get("parties", [])}
    exp_parties = {_norm(p["name"]) for p in gt.get("parties", [])}
    res["party_precision"], res["party_recall"] = _prf(pred_parties, exp_parties)

    pred_amt = {round(float(a.get("amount", 0)), 2) for a in record.get("monetary_amounts", [])}
    exp_amt = {round(float(a), 2) for a in gt.get("monetary_amounts", [])}
    res["amount_precision"], res["amount_recall"] = _prf(pred_amt, exp_amt)

    # --- Unsafe deadline accounting: the headline metric ---
    unsafe, label_notes = [], []
    expected = list(gt.get("deadlines", []))
    predicted = list(record.get("deadlines", []))
    matched_exp, matched_pred = set(), set()
    pair_map = {}  # predicted index -> expected index (keep the actual pairing)

    # Pass 1: greedy best-similarity pairing on the obligation text.
    pairs = sorted(
        ((_similarity(p.get("obligation"), e["obligation"]), pi, ei)
         for pi, p in enumerate(predicted) for ei, e in enumerate(expected)),
        reverse=True)
    for sim, pi, ei in pairs:
        if sim < 0.5 or pi in matched_pred or ei in matched_exp:
            continue
        matched_pred.add(pi)
        matched_exp.add(ei)
        pair_map[pi] = ei
        if sim < 1.0:
            label_notes.append(f"label variance: '{predicted[pi].get('obligation')}' "
                               f"~ '{expected[ei]['obligation']}'")

    # Pass 2: an unmatched prediction whose date equals an unmatched expected
    # date is the same deadline under another name, not a fabrication.
    for pi, p in enumerate(predicted):
        if pi in matched_pred or not p.get("computed_due_date"):
            continue
        for ei, e in enumerate(expected):
            if ei in matched_exp or e.get("computed_due_date") != p["computed_due_date"]:
                continue
            matched_pred.add(pi)
            matched_exp.add(ei)
            pair_map[pi] = ei
            label_notes.append(f"matched by date only: '{p.get('obligation')}' "
                               f"~ '{e['obligation']}'")
            break

    # Now score only what is genuinely wrong, using the pairing we computed.
    already_covered = {expected[ei].get("computed_due_date")
                       for ei in matched_exp if expected[ei].get("computed_due_date")}
    for pi, p in enumerate(predicted):
        if pi in matched_pred or not p.get("computed_due_date"):
            continue
        if p["computed_due_date"] in already_covered:
            # The model split one obligation into two entries carrying the same
            # date. That is phrasing, not a safety failure.
            label_notes.append(f"split obligation: '{p.get('obligation')}' shares a "
                               f"date with an already-matched deadline")
        else:
            unsafe.append(f"date produced for obligation not in ground truth "
                          f"'{p.get('obligation')}' -> {p['computed_due_date']}")

    for pi, ei in pair_map.items():
        exp_due = expected[ei].get("computed_due_date")
        got_due = predicted[pi].get("computed_due_date")
        if exp_due is None and got_due:
            unsafe.append(f"computed {got_due} where no date is justified "
                          f"({expected[ei]['obligation']})")
        elif exp_due and got_due != exp_due:
            unsafe.append(f"wrong date for {expected[ei]['obligation']}: "
                          f"got {got_due}, expected {exp_due}")

    for ei, e in enumerate(expected):
        if ei not in matched_exp:
            unsafe.append(f"missed deadline entirely: {e['obligation']}")

    res["label_variance"] = "; ".join(label_notes)
    exp_by_obl = expected

    res["unsafe_deadlines"] = len(unsafe)
    res["unsafe_detail"] = "; ".join(unsafe)
    res["deadline_count_expected"] = len(expected)

    # --- Provenance validity ---
    norm_src = re.sub(r"\s+", " ", source_text).strip().lower()
    total = valid = 0
    for key in ("parties", "dates", "deadlines", "obligations", "monetary_amounts"):
        for item in record.get(key, []) or []:
            q = (item.get("source") or {}).get("quote", "")
            total += 1
            if q and re.sub(r"\s+", " ", q).strip().lower() in norm_src:
                valid += 1
    res["provenance_valid_pct"] = round(100 * valid / total, 1) if total else 100.0

    # --- Flag recall ---
    raised = {f["type"] for f in record.get("flags", [])}
    expected_flags = set(gt.get("expected_flags", []))
    res["flags_missed"] = ",".join(sorted(expected_flags - raised))
    res["flag_recall"] = (round(len(expected_flags & raised) / len(expected_flags), 3)
                          if expected_flags else 1.0)

    meta = record.get("extraction_meta", {})
    res["cost_usd"] = meta.get("cost_usd", 0)
    res["latency_ms"] = meta.get("latency_ms", 0)
    res["tokens_in"] = meta.get("tokens_in", 0)
    res["tokens_out"] = meta.get("tokens_out", 0)
    res["extract_model"] = meta.get("extract_model", "")
    res["prompt_version"] = meta.get("prompt_version", "")
    return res
