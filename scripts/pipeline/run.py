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
from ..clustering import cluster
from ..nlp import word2vec as w2v_module
from ..preprocessing.preprocess import run_preprocessing

SPLITS = ["train", "val", "test"]


def run_nlp_features():
    """Run NLP feature extraction on all splits."""
    print("\n" + "=" * 60)
    print("STEP 2: NLP FEATURE EXTRACTION")
    print("=" * 60)
    nlp_extract.main()


def run_clustering():
    """Run clustering on NLP features for all splits."""
    print("\n" + "=" * 60)
    print("STEP 3: CLUSTERING")
    print("=" * 60)

    from scripts.db import get_engine
    engine = get_engine()

    w2v_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "word2vec.model"
    )
    if os.path.exists(w2v_path):
        w2v_model = w2v_module.load()
    else:
        print("  No Word2Vec model found, skipping keyword clustering")
        w2v_model = None

    all_results = {}
    for split in SPLITS:
        print(f"\n  --- Clustering {split.upper()} ---")

        if w2v_model is not None:
            kw_agg, kw_info = cluster.cluster_keyword_vectors_from_db(split, w2v_model, engine)
            if kw_agg is not None:
                with engine.connect() as conn:
                    from sqlalchemy import text
                    conn.execute(text(f"DROP TABLE IF EXISTS {split}_cluster_keyword"))
                    conn.commit()
                kw_agg.to_sql(f"{split}_cluster_keyword", engine, if_exists="append", index=False)
                print(f"    {split}_cluster_keyword: {len(kw_agg):,} rows")
                all_results[f"{split}_keyword"] = kw_info

        sent_agg, sent_info = cluster.cluster_sentiment_from_db(split, engine)
        if sent_agg is not None:
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text(f"DROP TABLE IF EXISTS {split}_cluster_sentiment"))
                conn.commit()
            sent_agg.to_sql(f"{split}_cluster_sentiment", engine, if_exists="append", index=False)
            print(f"    {split}_cluster_sentiment: {len(sent_agg):,} rows")
            all_results[f"{split}_sentiment"] = sent_info

    cluster.save_clustering_models({
        "keyword_results": all_results,
    })

    print(f"\n  Clustering complete")


def run_training(activation=None, reg_loss=None, epochs=None, lr=None, patience=None):
    """Run full model training."""
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)
    kwargs = {"verbose": True}
    # Only forward what was explicitly provided so train_all_models keeps its
    # own defaults otherwise.
    for key, val in (("activation", activation), ("reg_loss", reg_loss),
                     ("epochs", epochs), ("lr", lr), ("patience", patience)):
        if val is not None:
            kwargs[key] = val
    return train_all_models(**kwargs)


def run_prediction(results=None):
    """Run prediction on test set."""
    print("\n" + "=" * 60)
    print("STEP 5: PREDICTION")
    print("=" * 60)
    models = load_models()
    if not models:
        print("  No models found, skipping prediction")
        return
    result = predict_all(models)
    if result is None:
        print("  No prediction data available")
        return
    predictions, df_test = result
    if predictions:
        save_predictions_to_db(predictions, df_test)


def run_pipeline(skip_nlp=False, skip_clustering=False, activation=None,
                 reg_loss=None, epochs=None, lr=None, patience=None):
    """
    Full pipeline execution.

    Args:
        skip_nlp: skip NLP feature extraction (use if already done)
        skip_clustering: skip clustering step (use if already done)
        activation: activation for all neural models (default leaky_relu)
        reg_loss: regression loss for sequence models (huber/mse/mae)
        epochs / lr / patience: training overrides
    """
    start = time.time()

    print("=" * 60)
    print("CAF STOCK PREDICTION — FULL PIPELINE")
    print("=" * 60)

    # Step 1: Preprocessing (creates split tables)
    print("\n" + "=" * 60)
    print("STEP 1: PREPROCESSING")
    print("=" * 60)
    run_preprocessing()

    # Step 2: NLP features
    if not skip_nlp:
        run_nlp_features()
    else:
        print("\nSkipping NLP (skip_nlp=True)")

    # Step 2: Clustering
    if not skip_clustering:
        run_clustering()
    else:
        print("\nSkipping Clustering (skip_clustering=True)")

    # Step 3: Training
    results = run_training(activation=activation, reg_loss=reg_loss,
                           epochs=epochs, lr=lr, patience=patience)

    # Step 4: Prediction
    run_prediction(results)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAF Stock Prediction Pipeline")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP feature extraction")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering step")
    args = parser.parse_args()
    run_pipeline(skip_nlp=args.skip_nlp, skip_clustering=args.skip_clustering)
