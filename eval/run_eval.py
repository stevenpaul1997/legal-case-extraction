"""Run the pipeline over the test set and write a results row per document.

    python -m eval.run_eval --provider mock
    python -m eval.run_eval --provider groq --extract-model llama-3.3-70b-versatile
"""
import argparse
import csv
import json
import pathlib
import sys
import time
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.score import score  # noqa: E402
from src.pipeline import process  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "documents"
GT = ROOT / "data" / "ground_truth"
RESULTS = ROOT / "results"

FIELDS = ["doc_id", "extract_model", "prompt_version", "doc_type_correct",
          "party_precision", "party_recall", "amount_precision", "amount_recall",
          "unsafe_deadlines", "deadline_count_expected", "provenance_valid_pct",
          "flag_recall", "flags_missed", "cost_usd", "latency_ms",
          "tokens_in", "tokens_out", "unsafe_detail", "label_variance", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "groq"])
    ap.add_argument("--extract-model", default=None)
    ap.add_argument("--classify-model", default=None)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD, pins urgency checks")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rescore", action="store_true",
                    help="Re-score the saved records in results/records without "
                         "calling the API. Use after changing the scorer.")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="Seconds to pause between documents (free tiers rate limit)")
    args = ap.parse_args()

    today = date.fromisoformat(args.today) if args.today else None
    RESULTS.mkdir(exist_ok=True)
    out_path = pathlib.Path(args.out) if args.out else RESULTS / "eval_results.csv"
    records_dir = RESULTS / "records"
    records_dir.mkdir(exist_ok=True)

    rows = []
    for i, doc in enumerate(sorted(DOCS.glob("*.txt"))):
        if i and args.provider != "mock" and args.delay and not args.rescore:
            time.sleep(args.delay)
        gt_path = GT / f"{doc.stem}.json"
        if not gt_path.exists():
            print(f"  skip {doc.stem} (no ground truth)")
            continue
        gt = json.loads(gt_path.read_text())
        saved = records_dir / f"{doc.stem}.json"
        if args.rescore:
            if not saved.exists():
                print(f"  skip {doc.stem} (no saved record)")
                continue
            record = json.loads(saved.read_text())
            row = score(record, gt, doc.read_text(errors="replace"))
            row["notes"] = "rescored"
            rows.append(row)
            print(f"  {row['doc_id']:<26} unsafe={row['unsafe_deadlines']} "
                  f"prov={row['provenance_valid_pct']}% flagrecall={row['flag_recall']}")
            continue
        try:
            record = process(doc, provider_name=args.provider,
                             extract_model=args.extract_model,
                             classify_model=args.classify_model, today=today)
        except Exception as exc:  # a crash is a result too
            rows.append({"doc_id": doc.stem, "notes": f"ERROR: {exc}"})
            print(f"  {doc.stem}: ERROR {exc}")
            continue
        (records_dir / f"{doc.stem}.json").write_text(json.dumps(record, indent=2))
        row = score(record, gt, doc.read_text(errors="replace"))
        row["notes"] = ""
        rows.append(row)
        print(f"  {row['doc_id']:<26} unsafe={row['unsafe_deadlines']} "
              f"prov={row['provenance_valid_pct']}% flagrecall={row['flag_recall']} "
              f"${row['cost_usd']:.5f} {row['latency_ms']}ms")

    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    scored = [r for r in rows if "unsafe_deadlines" in r]
    n = len(scored) or 1
    total_deadlines = sum(r["deadline_count_expected"] for r in scored) or 1
    print("\n--- SUMMARY ---")
    print(f"documents scored        : {len(scored)}")
    print(f"doc_type accuracy       : {100*sum(r['doc_type_correct'] for r in scored)/n:.1f}%")
    print(f"unsafe deadlines        : {sum(r['unsafe_deadlines'] for r in scored)} "
          f"of {total_deadlines} expected deadlines")
    print(f"provenance valid        : {sum(r['provenance_valid_pct'] for r in scored)/n:.1f}%")
    print(f"flag recall             : {sum(r['flag_recall'] for r in scored)/n:.2f}")
    print(f"party recall            : {sum(r['party_recall'] for r in scored)/n:.2f}")
    print(f"cost per document       : ${sum(r['cost_usd'] for r in scored)/n:.5f}")
    print(f"mean latency            : {sum(r['latency_ms'] for r in scored)/n:.0f} ms")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
