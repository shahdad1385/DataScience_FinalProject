"""
Pipeline — full orchestrator.

Runs: NLP features → clustering → data assembly → training → prediction → save.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .train import train_all_models
from .predict import load_models, predict_all, save_predictions_to_db
from ..nlp import extract as nlp_extract


def run_nlp_features():
    """Run NLP feature extraction on all splits."""
    print("\n" + "=" * 60)
    print("STEP 1: NLP FEATURE EXTRACTION")
    print("=" * 60)
    nlp_extract.main()


def run_training():
    """Run full model training."""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    return train_all_models(verbose=True)


def run_prediction(results=None):
    """Run prediction on test set."""
    print("\n" + "=" * 60)
    print("STEP 3: PREDICTION")
    print("=" * 60)
    models = load_models()
    predictions, df_test = predict_all(models)
    if predictions:
        save_predictions_to_db(predictions, df_test)


def run_pipeline(skip_nlp=False):
    """
    Full pipeline execution.

    Args:
        skip_nlp: skip NLP feature extraction (use if already done)
    """
    start = time.time()

    print("=" * 60)
    print("CAF STOCK PREDICTION — FULL PIPELINE")
    print("=" * 60)

    # Step 1: NLP features
    if not skip_nlp:
        run_nlp_features()
    else:
        print("\nSkipping NLP (skip_nlp=True)")

    # Step 2: Training
    results = run_training()

    # Step 3: Prediction
    run_prediction(results)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAF Stock Prediction Pipeline")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP feature extraction")
    args = parser.parse_args()
    run_pipeline(skip_nlp=args.skip_nlp)
