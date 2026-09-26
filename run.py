"""Process one document from the command line.

    python run.py data/documents/03_summons_no_rule.txt --provider mock
"""
import argparse
import json

from src.pipeline import process

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--provider", default="mock", choices=["mock", "groq"])
ap.add_argument("--extract-model", default=None)
args = ap.parse_args()

print(json.dumps(process(args.path, provider_name=args.provider,
                         extract_model=args.extract_model), indent=2))
