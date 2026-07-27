"""
CAF Stock Prediction — Main Entry Point.

Usage:
    python pipeline.py --mode train              # Train all models from scratch
    python pipeline.py --mode finetune           # Hyperparameter tuning with Optuna
    python pipeline.py --mode evaluate           # Evaluate saved models on test set
    python pipeline.py --mode full               # Preprocess + train + evaluate
    python pipeline.py --mode train --model xgboost   # Train only XGBoost
    python pipeline.py --mode evaluate --model lstm    # Evaluate only LSTM
    python pipeline.py --mode finetune --trials 50     # 50 Optuna trials
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.pipeline.run import run_pipeline
from scripts.pipeline.train import train_all_models, evaluate_saved_models
from scripts.pipeline.finetune import finetune_all


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CAF Stock Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train      Train all models from scratch, save weights
  finetune   Hyperparameter tuning with Optuna on validation set
  evaluate   Load saved weights and evaluate on test set
  full       Preprocess data, then train, then evaluate

Examples:
  python pipeline.py --mode train
  python pipeline.py --mode finetune --trials 100
  python pipeline.py --mode evaluate --model xgboost
  python pipeline.py --mode full
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "finetune", "evaluate", "full"],
        help="Pipeline mode (default: train)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["lstm", "gru", "transformer", "bilstm", "tcn",
                 "xgboost", "lightgbm", "random_forest", "mlp", "ridge", "logistic"],
        help="Run only this model (default: all)",
    )
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials for finetune mode")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP feature extraction")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    if args.mode == "full":
        from scripts.preprocessing.load_data import create_database
        from scripts.preprocessing.feature_engineering import (
            add_stock_features, add_indicator_features, add_news_features,
        )
        from scripts.preprocessing.preprocess import run_preprocessing
        print("=" * 60)
        print("FULL MODE: Preprocessing + Training + Evaluation")
        print("=" * 60)
        create_database()
        add_stock_features()
        add_indicator_features()
        add_news_features()
        run_preprocessing()

    if args.mode in ("train", "full"):
        train_all_models(
            verbose=args.verbose,
            model_filter=args.model,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            skip_nlp=args.skip_nlp,
        )
        evaluate_saved_models(model_filter=args.model)

    elif args.mode == "finetune":
        finetune_all(
            trials=args.trials,
            model_filter=args.model,
            verbose=args.verbose,
            skip_nlp=args.skip_nlp,
        )

    elif args.mode == "evaluate":
        evaluate_saved_models(model_filter=args.model)


if __name__ == "__main__":
    main()
