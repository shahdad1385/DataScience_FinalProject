"""
CAF Stock Prediction — Entry Point.

Run: python run_pipeline.py [--skip-nlp]
"""

from scripts.pipeline.run import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAF Stock Prediction Pipeline")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP feature extraction")
    args = parser.parse_args()
    run_pipeline(skip_nlp=args.skip_nlp)
