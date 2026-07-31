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

Evaluation flags:
    --eval-detailed      Per-ticker & per-output metrics breakdown
    --eval-confusion     Save confusion matrices as PNG + CSV
    --eval-trading       Trading metrics: Sharpe, Sortino, hit rate, P&L
    --eval-feature-imp   Feature importance for tree models
    --eval-statistical   Diebold-Mariano test, permutation importance
    --eval-calibration   Reliability diagrams for classifiers
    --eval-save-preds    Save raw predictions to CSV
    --eval-save-plots    Save all plots to models/eval_plots/
    --eval-ensemble      Evaluate ensemble predictions
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.pipeline.run import run_pipeline
from scripts.pipeline.train import train_all_models, evaluate_saved_models
from scripts.pipeline.finetune import finetune_all
from scripts.activations import ACTIVATION_CHOICES, DEFAULT_ACTIVATION
from scripts.timeseries.train import DEFAULT_REG_LOSS
from scripts.pipeline.data_assembly import HORIZON, SEQ_LEN


def _patch_defaults(fn, **overrides):
    """Rebind keyword defaults captured in a function signature.

    Functions like `prepare_data(split, feature_cols=None, horizon=HORIZON)`
    evaluate HORIZON once, at definition time. Rebinding the module constant
    afterwards does not touch that captured default, so a --horizon override
    would set da.HORIZON while prepare_data kept building 5-day targets. This
    rewrites __defaults__ in place so the flag actually reaches the targets.
    """
    defaults = list(fn.__defaults__ or ())
    if not defaults:
        return
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    offset = len(names) - len(defaults)
    for i, nm in enumerate(names[offset:]):
        if nm in overrides:
            defaults[i] = overrides[nm]
    fn.__defaults__ = tuple(defaults)


def _apply_data_overrides(horizon=None, seq_len=None):
    """Propagate --horizon / --seq-len to every module that holds a copy.

    data_assembly defines HORIZON and SEQ_LEN, but train.py, predict.py,
    evaluate.py and finetune.py all do `from .data_assembly import SEQ_LEN`,
    which binds the value at import time. Rebinding only the source module would
    leave those copies stale, so the override is pushed to each holder, and to
    the default arguments of every function that captured either constant.
    """
    from scripts.pipeline import data_assembly as da

    horizon = int(horizon) if horizon else da.HORIZON
    seq_len = int(seq_len) if seq_len else da.SEQ_LEN
    if horizon < 1 or seq_len < 2:
        raise SystemExit(f"Invalid --horizon {horizon} / --seq-len {seq_len}")

    da.HORIZON = horizon
    da.SEQ_LEN = seq_len

    # Function defaults captured at definition time.
    _patch_defaults(da.create_sequences, seq_len=seq_len)
    _patch_defaults(da.create_sequences_with_context, seq_len=seq_len)
    _patch_defaults(da.prepare_data, horizon=horizon)
    _patch_defaults(da.add_targets, horizon=horizon)
    _patch_defaults(da.dead_zone_for, horizon=horizon)

    # Modules that imported the constants by value.
    for mod_name in ("scripts.pipeline.train", "scripts.pipeline.predict",
                     "scripts.pipeline.evaluate", "scripts.pipeline.finetune"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "SEQ_LEN"):
            mod.SEQ_LEN = seq_len
        if hasattr(mod, "HORIZON"):
            mod.HORIZON = horizon
        for fn_name in ("evaluate_trading", "_actual_forward_returns"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                _patch_defaults(fn, horizon=horizon)

    if (horizon, seq_len) != (HORIZON, SEQ_LEN):
        print(f"Data config: horizon={horizon}d, lookback={seq_len}d "
              f"(defaults {HORIZON}d / {SEQ_LEN}d)")
    return horizon, seq_len


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
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering step")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument(
        "--activation", type=str, default=DEFAULT_ACTIVATION,
        choices=ACTIVATION_CHOICES,
        help=f"Activation for all neural models (default: {DEFAULT_ACTIVATION})",
    )
    parser.add_argument(
        "--reg-loss", type=str, default=DEFAULT_REG_LOSS,
        choices=["huber", "mse", "mae"],
        help=f"Regression loss for sequence models (default: {DEFAULT_REG_LOSS}); "
             "huber is robust to the fat tails in daily returns",
    )
    parser.add_argument(
        "--horizon", type=int, default=HORIZON,
        help=f"Forward trading days to predict (default: {HORIZON}). "
             "1-day direction is near-random for NVDA (lag-1 autocorr -0.08); "
             "5d has ~2.3x the signal-to-noise, 21d ~4.6x.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=SEQ_LEN,
        help=f"Lookback window in trading days (default: {SEQ_LEN} ~= 3 months). "
             "21 ~= 1 month, 63 ~= 3 months, 126 ~= 6 months.",
    )

    # Evaluation flags
    parser.add_argument("--eval-detailed", action="store_true", help="Per-ticker & per-output metrics breakdown")
    parser.add_argument("--eval-confusion", action="store_true", help="Save confusion matrices as PNG + CSV")
    parser.add_argument("--eval-trading", action="store_true", help="Trading metrics: Sharpe, Sortino, hit rate, P&L")
    parser.add_argument("--eval-feature-imp", action="store_true", help="Feature importance for tree models")
    parser.add_argument("--eval-statistical", action="store_true", help="Diebold-Mariano test, permutation importance")
    parser.add_argument("--eval-calibration", action="store_true", help="Reliability diagrams for classifiers")
    parser.add_argument("--eval-save-preds", action="store_true", help="Save raw predictions to CSV")
    parser.add_argument("--eval-save-plots", action="store_true", help="Save all plots to models/eval_plots/")
    parser.add_argument("--eval-ensemble", action="store_true", help="Evaluate ensemble predictions")

    args = parser.parse_args()

    eval_flags = {
        "detailed": args.eval_detailed,
        "confusion": args.eval_confusion,
        "trading": args.eval_trading,
        "feature_imp": args.eval_feature_imp,
        "statistical": args.eval_statistical,
        "calibration": args.eval_calibration,
        "save_preds": args.eval_save_preds,
        "save_plots": args.eval_save_plots,
        "ensemble": args.eval_ensemble,
    }

    if args.mode == "full":
        from scripts.preprocessing.load_data import create_database
        # All six feature builders must run. The three sentiment/social/event
        # builders were previously omitted here, so a from-scratch `--mode full`
        # rebuild left those columns NULL and the DB only ever had them because
        # they had been run by hand.
        from scripts.preprocessing.feature_engineering import (
            add_stock_features, add_indicator_features, add_news_features,
            add_sentiment_features, add_social_sentiment_features,
            add_economic_event_features,
        )
        from scripts.preprocessing.preprocess import run_preprocessing
        print("=" * 60)
        print("FULL MODE: Preprocessing + Training + Evaluation")
        print("=" * 60)
        create_database()
        add_stock_features()
        add_indicator_features()
        add_news_features()
        add_sentiment_features()
        add_social_sentiment_features()
        add_economic_event_features()
        run_preprocessing()

    # Apply horizon / lookback overrides before any data is assembled. These are
    # module-level constants that data_assembly reads at call time, so setting
    # them here propagates to train, finetune, predict and evaluate alike.
    _apply_data_overrides(horizon=args.horizon, seq_len=args.seq_len)

    if args.mode in ("train", "full"):
        run_pipeline(
            skip_nlp=args.skip_nlp, skip_clustering=args.skip_clustering,
            activation=args.activation, reg_loss=args.reg_loss,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
        )
        evaluate_saved_models(model_filter=args.model, eval_flags=eval_flags)

    elif args.mode == "finetune":
        finetune_all(
            trials=args.trials,
            model_filter=args.model,
            verbose=args.verbose,
            skip_nlp=args.skip_nlp,
        )

    elif args.mode == "evaluate":
        evaluate_saved_models(model_filter=args.model, eval_flags=eval_flags)


if __name__ == "__main__":
    main()
