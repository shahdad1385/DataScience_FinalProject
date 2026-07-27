"""
Pipeline — train all models, compare, ensemble, select best.

Orchestrates: NLP features → clustering → data assembly → time series models → tabular models → ensemble.
"""

import os
import sys
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_assembly import (
    prepare_data, prepare_sequences, flatten_sequences, TICKERS, SEQ_LEN,
)

# Import all model modules
from ..timeseries import pipe as ts_pipe
from ..tabular import pipe as tab_pipe
from ..clustering import cluster


def log_mlflow_metrics(results, prefix):
    """Log all model metrics from a results dict to MLflow."""
    for name, res in results.items():
        if name.startswith("_"):
            continue
        overall = res.get("overall", {})
        for metric_name, value in overall.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{prefix}_{name}_{metric_name}", value)


def train_all_models(verbose=True):
    """
    Full training pipeline.

    Steps:
    1. Load and assemble data
    2. Train time series models (LSTM, GRU, Transformer, BiLSTM, TCN)
    3. Train tabular models (XGBoost, RF, LightGBM, MLP, Logistic, Ridge)
    4. Compare and select best
    5. Ensemble voting
    6. Save results

    Returns:
        dict with all results
    """
    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)

    mlflow.set_experiment("CAF_Stock_Prediction")
    run = mlflow.start_run(run_name="full_pipeline")
    try:
        # === 1. Load and prepare data ===
        print("\n[1/6] Loading data...")
        X_train, y_reg_train, y_cls_train, feature_names, df_train = prepare_data("train")
        X_val, y_reg_val, y_cls_val, _, df_val = prepare_data("val")
        X_test, y_reg_test, y_cls_test, _, df_test = prepare_data("test")

        if X_train is None:
            print("ERROR: No training data.")
            return None

        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_val", X_val.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])

        # === 2. Prepare sequences ===
        print("\n[2/6] Preparing sequences...")
        (X_tr_seq, y_reg_tr, y_cls_tr,
         X_v_seq, y_reg_v, y_cls_v,
         X_te_seq, y_reg_te, y_cls_te) = prepare_sequences(
            X_train, y_reg_train, y_cls_train,
            X_val, y_reg_val, y_cls_val,
            X_test, y_reg_test, y_cls_test,
        )

        X_tr_flat = flatten_sequences(X_tr_seq)
        X_v_flat = flatten_sequences(X_v_seq)
        X_te_flat = flatten_sequences(X_te_seq)

        input_size = X_tr_seq.shape[2]
        n_tickers = len(TICKERS)

        mlflow.log_param("seq_len", SEQ_LEN)
        mlflow.log_param("n_tickers", n_tickers)
        mlflow.log_param("n_features", input_size)
        mlflow.log_param("tickers", str(TICKERS))

        # === 3. Train time series models ===
        print("\n[3/6] Training time series models...")
        ts_results = ts_pipe.train_all(
            X_tr_seq, y_reg_tr, y_cls_tr,
            X_v_seq, y_reg_v, y_cls_v,
            seq_len=SEQ_LEN, n_tickers=n_tickers, verbose=verbose,
        )
        log_mlflow_metrics(ts_results, "ts")

        # === 4. Train tabular models ===
        print("\n[4/6] Training tabular regression models...")
        tab_reg_results = tab_pipe.train_all_regression(
            X_tr_flat, y_reg_tr, X_v_flat, y_reg_v,
            feature_names=feature_names, verbose=verbose,
        )
        log_mlflow_metrics(tab_reg_results, "tab_reg")

        print("\n[4/6] Training tabular classification models...")
        tab_cls_results = tab_pipe.train_all_classification(
            X_tr_flat, y_cls_tr, X_v_flat, y_cls_v,
            feature_names=feature_names, verbose=verbose,
        )
        log_mlflow_metrics(tab_cls_results, "tab_cls")

        # === 5. Select best models ===
        print("\n[5/6] Selecting best models...")

        ts_best_name, ts_best = ts_pipe.select_best(ts_results)
        print(f"  Best time series: {ts_best_name} (val_loss={ts_best['val_loss']:.6f})")

        tab_reg_best = tab_reg_results["_best"]
        tab_reg_rmse = tab_reg_results[tab_reg_best]["overall"]["rmse"]
        print(f"  Best tabular regression: {tab_reg_best} (RMSE={tab_reg_rmse:.6f})")

        tab_cls_best = tab_cls_results["_best"]
        tab_cls_f1 = tab_cls_results[tab_cls_best]["overall"].get("f1", 0)
        print(f"  Best tabular classification: {tab_cls_best} (F1={tab_cls_f1:.4f})")

        # === 6. Ensemble voting ===
        print("\n[6/6] Ensemble voting...")
        ensemble = create_ensemble(
            ts_results, ts_best_name,
            tab_reg_results, tab_reg_best,
            tab_cls_results, tab_cls_best,
            X_v_seq, X_v_flat,
        )

        # Log best model info
        mlflow.log_param("ts_best_model", ts_best_name)
        mlflow.log_param("tab_reg_best_model", tab_reg_best)
        mlflow.log_param("tab_cls_best_model", tab_cls_best)
        mlflow.log_metric("ts_best_val_loss", ts_best["val_loss"])
        mlflow.log_metric("tab_reg_best_rmse", tab_reg_rmse)
        mlflow.log_metric("tab_cls_best_f1", tab_cls_f1)

        # Log feature names as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(feature_names))
            mlflow.log_artifact(f.name, "feature_names")
            os.unlink(f.name)

        # Log ensemble predictions as artifact
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(ensemble, f)
            mlflow.log_artifact(f.name, "ensemble")
            os.unlink(f.name)

        # === Save everything ===
        print("\nSaving results...")
        save_results(ts_results, tab_reg_results, tab_cls_results, ensemble,
                     ts_best_name, tab_reg_best, tab_cls_best, feature_names)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return {
            "ts_results": ts_results,
            "tab_reg_results": tab_reg_results,
            "tab_cls_results": tab_cls_results,
            "ensemble": ensemble,
            "ts_best": ts_best_name,
            "tab_reg_best": tab_reg_best,
            "tab_cls_best": tab_cls_best,
        }
    finally:
        mlflow.end_run()


def create_ensemble(ts_results, ts_best_name,
                    tab_reg_results, tab_reg_best,
                    tab_cls_results, tab_cls_best,
                    X_v_seq, X_v_flat):
    """
    Ensemble voting: combine predictions from top models.

    For regression: weighted average of top models' predictions.
    For classification: majority voting.
    """
    n_tickers = len(TICKERS)

    # Collect regression predictions
    reg_preds = []

    # Time series best
    ts_model = ts_results[ts_best_name]["model"]
    from ..timeseries.train import predict as ts_predict
    ts_loader = _make_loader(X_v_seq)
    ts_reg, _ = ts_predict(ts_model, ts_loader)
    reg_preds.append(("ts", ts_reg))

    # Tabular regression best
    tab_model = tab_reg_results[tab_reg_best]["model"]
    if tab_reg_best == "mlp":
        from ..tabular.mlp import predict_regressor as tab_predict
    elif tab_reg_best in ("xgboost", "lightgbm", "random_forest"):
        mod = {"xgboost": "xgboost_model", "lightgbm": "lightgbm_model", "random_forest": "random_forest"}[tab_reg_best]
        tab_predict = getattr(__import__(f"scripts.tabular.{mod}", fromlist=["predict_regressor"]), "predict_regressor")
    else:
        from ..tabular.ridge import predict as tab_predict
    tab_reg = tab_predict(tab_model, X_v_flat)
    reg_preds.append(("tab", tab_reg))

    # Weighted average (time series gets more weight)
    weights = [0.6, 0.4]  # ts, tab
    ensemble_reg = sum(w * p for w, (_, p) in zip(weights, reg_preds))

    # Collect classification predictions
    cls_preds = []

    # Time series best
    ts_cls = _[1]  # from ts_predict above
    cls_preds.append(("ts", ts_cls))

    # Tabular classification best
    tab_cls_model = tab_cls_results[tab_cls_best]["model"]
    if tab_cls_best == "mlp":
        from ..tabular.mlp import predict_classifier as tab_cls_predict
    elif tab_cls_best in ("xgboost", "lightgbm", "random_forest"):
        mod = {"xgboost": "xgboost_model", "lightgbm": "lightgbm_model", "random_forest": "random_forest"}[tab_cls_best]
        tab_cls_predict = getattr(__import__(f"scripts.tabular.{mod}", fromlist=["predict_classifier"]), "predict_classifier")
    elif tab_cls_best == "logistic":
        from ..tabular import logistic
        n_t = len(tab_cls_model)
        tab_cls_pred = np.zeros((X_v_flat.shape[0], n_t))
        tab_cls_prob = np.zeros((X_v_flat.shape[0], n_t))
        for t, lr_m in tab_cls_model.items():
            tab_cls_pred[:, t], tab_cls_prob[:, t] = logistic.predict(lr_m, X_v_flat)
        tab_cls_predict = None  # handled above
    else:
        tab_cls_predict = None

    if tab_cls_predict is not None:
        tab_cls_pred, tab_cls_prob = tab_cls_predict(tab_cls_model, X_v_flat)
    cls_preds.append(("tab", tab_cls_pred))

    # Majority voting
    ensemble_cls = (cls_preds[0][1] + cls_preds[1][1] > 1).astype(int)

    return {
        "regression": ensemble_reg,
        "classification": ensemble_cls,
    }


def _make_loader(X_seq):
    """Create a DataLoader from sequence data."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.FloatTensor(X_seq), torch.zeros(len(X_seq)), torch.zeros(len(X_seq)))
    return DataLoader(ds, batch_size=64, shuffle=False)


def save_results(ts_results, tab_reg_results, tab_cls_results, ensemble,
                 ts_best, tab_reg_best, tab_cls_best, feature_names):
    """Save all results to disk."""
    from ..timeseries.train import save_model as save_ts

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(results_dir, exist_ok=True)

    # Save best time series model
    ts_model = ts_results[ts_best]["model"]
    save_ts(ts_model, f"timeseries_{ts_best}", ts_results[ts_best]["model"].reg_head.in_features)

    # Save tabular models
    for name, res in tab_reg_results.items():
        if name.startswith("_"):
            continue
        model = res["model"]
        if name in ("xgboost", "lightgbm", "random_forest"):
            import pickle
            with open(os.path.join(results_dir, f"tabular_{name}_reg.pkl"), "wb") as f:
                pickle.dump(model, f)
        elif name == "mlp":
            from ..tabular.mlp import save_model as save_mlp
            save_mlp(model, f"tabular_{name}_reg", model.net[0].in_features,
                     [m.out_features for m in model.net if hasattr(m, 'out_features')], 20)
        elif name == "ridge":
            import pickle
            with open(os.path.join(results_dir, f"tabular_{name}_reg.pkl"), "wb") as f:
                pickle.dump(model, f)

    for name, res in tab_cls_results.items():
        if name.startswith("_"):
            continue
        model = res["model"]
        if name in ("xgboost", "lightgbm", "random_forest"):
            import pickle
            with open(os.path.join(results_dir, f"tabular_{name}_cls.pkl"), "wb") as f:
                pickle.dump(model, f)
        elif name == "mlp":
            from ..tabular.mlp import save_model as save_mlp
            save_mlp(model, f"tabular_{name}_cls", model.net[0].in_features,
                     [m.out_features for m in model.net if hasattr(m, 'out_features')], 5, is_classifier=True)
        elif name == "logistic":
            import pickle
            with open(os.path.join(results_dir, f"tabular_{name}_cls.pkl"), "wb") as f:
                pickle.dump(model, f)

    # Save feature names
    with open(os.path.join(results_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    # Save ensemble
    import pickle
    with open(os.path.join(results_dir, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)

    print(f"  All models saved to {results_dir}")
