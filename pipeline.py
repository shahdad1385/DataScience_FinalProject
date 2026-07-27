"""
CAF Stock Prediction — Main Entry Point.

Runs: NLP features → clustering → data assembly → training → prediction → save.

Usage:
    python pipeline.py [--skip-nlp]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.pipeline.run import run_pipeline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAF Stock Prediction Pipeline")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP feature extraction")
    args = parser.parse_args()
    run_pipeline(skip_nlp=args.skip_nlp)
