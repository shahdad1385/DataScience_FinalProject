"""
Pipeline — train all models, compare, ensemble, select best, evaluate.

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
from ..timeseries.train import DEFAULT_REG_LOSS
from ..tabular import pipe as tab_pipe
from ..clustering import cluster
from ..activations import DEFAULT_ACTIVATION

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def log_mlflow_metrics(results, prefix):
    """Log all model metrics from a results dict to MLflow."""
    for name, res in results.items():
        if name.startswith("_"):
            continue
        overall = res.get("overall", {})
        for metric_name, value in overall.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{prefix}_{name}_{metric_name}", value)


def load_hyperparams():
    """Load best hyperparameters from finetune run if available."""
    path = os.path.join(MODELS_DIR, "best_hyperparams.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def train_all_models(verbose=True, model_filter=None, epochs=200, lr=1e-3,
                     patience=30, skip_nlp=False, activation=DEFAULT_ACTIVATION,
                     reg_loss=DEFAULT_REG_LOSS):
    """
    Full training pipeline.

    Args:
        verbose: print progress
        model_filter: if set, only train this model (e.g. "xgboost", "lstm")
        epochs: max training epochs for time series models
        lr: learning rate for time series models
        patience: early stopping patience
        skip_nlp: skip NLP feature extraction
        activation: activation for all neural models (default leaky_relu)
        reg_loss: regression loss for sequence models (huber/mse/mae)
    """
    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)

    hyperparams = load_hyperparams()
    if hyperparams:
        print(f"Loaded hyperparameters from finetune run: {len(hyperparams)} models")

    mlflow.set_experiment("CAF_Stock_Prediction")
    run = mlflow.start_run(run_name="full_pipeline")
    try:
        # === 1. Load and prepare data ===
        print("\n[1/6] Loading data...")
        X_train, y_reg_train, y_cls_train, feature_names, df_train = prepare_data("train")
        X_val, y_reg_val, y_cls_val, _, df_val = prepare_data("val", feature_cols=feature_names)
        X_test, y_reg_test, y_cls_test, _, df_test = prepare_data("test", feature_cols=feature_names)

        if X_train is None:
            print("ERROR: No training data.")
            return None

        # Fit and apply a StandardScaler over the full joint feature matrix so
        # the unscaled NLP and embedding columns (Word2Vec, BERT, TF-IDF, keyword
        # counts) get normalized alongside the price/return/sentiment columns that
        # preprocess.py already scaled. Logistic hung before because it saw features
        # spanning 5+ orders of magnitude, and every other gradient/distance model
        # (MLP, XGBoost margins, etc.) also benefits from uniform scale.
        from .data_assembly import (
            fit_feature_scaler, apply_feature_scaler,
            fit_target_scaler, apply_target_scaler,
        )
        scaler = fit_feature_scaler(X_train)
        X_train = apply_feature_scaler(X_train, scaler)
        X_val = apply_feature_scaler(X_val, scaler)
        X_test = apply_feature_scaler(X_test, scaler)
        print("  Feature scaler fitted and applied to all splits")

        # Return targets are ~0.02 in magnitude, so an unscaled regression loss
        # is ~1e-4 next to a ~0.69 BCE term and the regression head would barely
        # train. Standardise the targets (train-fit only) so the two loss terms
        # are comparable; predictions are inverse-transformed before use.
        target_scaler = fit_target_scaler(y_reg_train)
        y_reg_train = apply_target_scaler(y_reg_train, target_scaler)
        y_reg_val = apply_target_scaler(y_reg_val, target_scaler)
        y_reg_test = apply_target_scaler(y_reg_test, target_scaler)
        print("  Target scaler fitted (returns standardised for balanced loss)")

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
        ts_models_to_train = ["lstm", "gru", "transformer", "bilstm", "tcn"]
        if model_filter and model_filter in ts_models_to_train:
            ts_models_to_train = [model_filter]
        elif model_filter and model_filter not in ts_models_to_train:
            ts_models_to_train = []

        # Loss configuration shared by every sequence model: Huber on the return
        # targets plus per-ticker BCE class balancing computed on train only.
        ts_loss_kwargs = ts_pipe.make_loss_kwargs(y_cls_tr, reg_loss=reg_loss)
        mlflow.log_param("activation", activation)
        mlflow.log_param("reg_loss", reg_loss)
        mlflow.log_param("target_type", "next_day_return")

        ts_results = {}
        if ts_models_to_train:
            print("\n[3/6] Training time series models...")
            for model_name in ts_models_to_train:
                hp = hyperparams.get(f"ts_{model_name}", {})
                model_lr = hp.get("lr", lr)
                model_epochs = hp.get("epochs", epochs)
                model_patience = hp.get("patience", patience)
                model_batch_size = hp.get("batch_size", 64)
                model_hidden = hp.get("hidden_size", 64)
                model_n_layers = hp.get("n_layers", 2)
                model_dropout = hp.get("dropout", 0.2)

                print(f"\n  Training {model_name.upper()} (lr={model_lr:.2e}, hidden={model_hidden}, "
                      f"layers={model_n_layers}, act={activation}, reg_loss={reg_loss})")
                result = ts_pipe.train_single(
                    model_name,
                    X_tr_seq, y_reg_tr, y_cls_tr,
                    X_v_seq, y_reg_v, y_cls_v,
                    seq_len=SEQ_LEN, n_tickers=n_tickers,
                    lr=model_lr, epochs=model_epochs, patience=model_patience,
                    batch_size=model_batch_size, hidden_size=model_hidden,
                    n_layers=model_n_layers, dropout=model_dropout,
                    verbose=verbose,
                    activation=hp.get("activation", activation),
                    loss_kwargs=ts_loss_kwargs,
                )
                ts_results[model_name] = result
            log_mlflow_metrics(ts_results, "ts")

        # === 4. Train tabular models ===
        tab_reg_models_to_train = ["xgboost", "random_forest", "mlp", "ridge"]
        tab_cls_models_to_train = ["xgboost", "random_forest", "mlp", "logistic"]
        try:
            import lightgbm
            tab_reg_models_to_train.insert(2, "lightgbm")
            tab_cls_models_to_train.insert(2, "lightgbm")
        except ImportError:
            pass

        if model_filter:
            if model_filter in tab_reg_models_to_train:
                tab_reg_models_to_train = [model_filter]
            elif model_filter not in ("lstm", "gru", "transformer", "bilstm", "tcn"):
                tab_reg_models_to_train = []
            else:
                tab_reg_models_to_train = []

            if model_filter in tab_cls_models_to_train:
                tab_cls_models_to_train = [model_filter]
            elif model_filter not in ("lstm", "gru", "transformer", "bilstm", "tcn"):
                tab_cls_models_to_train = []
            else:
                tab_cls_models_to_train = []

        tab_reg_results = {}
        if tab_reg_models_to_train:
            print("\n[4/6] Training tabular regression models...")
            for model_name in tab_reg_models_to_train:
                hp = hyperparams.get(f"tab_reg_{model_name}", {})
                print(f"\n  Training {model_name.upper()} regression (params: {hp})")
            tab_reg_results = tab_pipe.train_all_regression(
                X_tr_flat, y_reg_tr, X_v_flat, y_reg_v,
                feature_names=feature_names, verbose=verbose,
                model_filter=tab_reg_models_to_train, hyperparams=hyperparams,
                activation=activation,
            )
            log_mlflow_metrics(tab_reg_results, "tab_reg")

        tab_cls_results = {}
        if tab_cls_models_to_train:
            print("\n[4/6] Training tabular classification models...")
            for model_name in tab_cls_models_to_train:
                hp = hyperparams.get(f"tab_cls_{model_name}", {})
                print(f"\n  Training {model_name.upper()} classification (params: {hp})")
            tab_cls_results = tab_pipe.train_all_classification(
                X_tr_flat, y_cls_tr, X_v_flat, y_cls_v,
                feature_names=feature_names, verbose=verbose,
                model_filter=tab_cls_models_to_train, hyperparams=hyperparams,
                activation=activation,
            )
            log_mlflow_metrics(tab_cls_results, "tab_cls")

        # === 5. Select best models ===
        print("\n[5/6] Selecting best models...")

        ts_best_name, ts_best = None, None
        if ts_results:
            ts_best_name, ts_best = ts_pipe.select_best(ts_results)
            print(f"  Best time series: {ts_best_name} (val_loss={ts_best['val_loss']:.6f})")

        tab_reg_best, tab_reg_rmse = None, None
        if tab_reg_results and "_best" in tab_reg_results:
            tab_reg_best = tab_reg_results["_best"]
            tab_reg_rmse = tab_reg_results[tab_reg_best]["overall"]["rmse"]
            print(f"  Best tabular regression: {tab_reg_best} (RMSE={tab_reg_rmse:.6f})")

        tab_cls_best, tab_cls_f1 = None, None
        if tab_cls_results and "_best" in tab_cls_results:
            tab_cls_best = tab_cls_results["_best"]
            tab_cls_f1 = tab_cls_results[tab_cls_best]["overall"].get("f1", 0)
            print(f"  Best tabular classification: {tab_cls_best} (F1={tab_cls_f1:.4f})")

        # === 6. Ensemble voting ===
        ensemble = None
        if ts_results and tab_reg_results and tab_cls_results:
            print("\n[6/6] Ensemble voting...")
            ensemble = create_ensemble(
                ts_results, ts_best_name,
                tab_reg_results, tab_reg_best,
                tab_cls_results, tab_cls_best,
                X_v_seq, X_v_flat,
            )
        else:
            print("\n[6/6] Skipping ensemble (not all model types trained)")

        # Log best model info
        if ts_best_name:
            mlflow.log_param("ts_best_model", ts_best_name)
            mlflow.log_metric("ts_best_val_loss", ts_best["val_loss"])
        if tab_reg_best:
            mlflow.log_param("tab_reg_best_model", tab_reg_best)
            mlflow.log_metric("tab_reg_best_rmse", tab_reg_rmse)
        if tab_cls_best:
            mlflow.log_param("tab_cls_best_model", tab_cls_best)
            mlflow.log_metric("tab_cls_best_f1", tab_cls_f1)

        # Log feature names as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(feature_names))
            mlflow.log_artifact(f.name, "feature_names")
            os.unlink(f.name)

        # Log ensemble predictions as artifact
        if ensemble:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(ensemble, f)
                mlflow.log_artifact(f.name, "ensemble")
                os.unlink(f.name)

        # === Save everything ===
        print("\nSaving results...")
        save_results(ts_results, tab_reg_results, tab_cls_results, ensemble,
                     ts_best_name, tab_reg_best, tab_cls_best, feature_names)

        # Save best model selection JSON
        selection = save_best_model_selection(
            ts_best_name, ts_results,
            tab_reg_best, tab_reg_results,
            tab_cls_best, tab_cls_results,
            feature_names,
        )

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


def evaluate_saved_models(model_filter=None, eval_flags=None):
    """
    Load saved models and evaluate on test set.

    Uses best_model_selection.json to load only the best model per category.
    Falls back to loading all models if JSON not found.

    Args:
        model_filter: if set, only evaluate this model
        eval_flags: dict of evaluation flags (detailed, confusion, trading, etc.)
    """
    if eval_flags is None:
        eval_flags = {}
    print("\n" + "=" * 60)
    print("EVALUATION MODE — Loading saved models")
    print("=" * 60)
    if any(eval_flags.values()):
        print(f"  Evaluation flags enabled: {[k for k, v in eval_flags.items() if v]}")

    # Load best model selection
    selection = load_best_model_selection()
    if selection:
        print(f"  Loaded best model selection from JSON")
        ts_best_name = selection.get("timeseries", {}).get("best_model")
        tab_reg_best_name = selection.get("tabular_regression", {}).get("best_model")
        tab_cls_best_name = selection.get("tabular_classification", {}).get("best_model")
        if ts_best_name:
            print(f"    Best time series: {ts_best_name}")
        if tab_reg_best_name:
            print(f"    Best tabular regression: {tab_reg_best_name}")
        if tab_cls_best_name:
            print(f"    Best tabular classification: {tab_cls_best_name}")
    else:
        print(f"  No best_model_selection.json found, loading all available models")
        ts_best_name = None
        tab_reg_best_name = None
        tab_cls_best_name = None

    # Load test data
    X_train, y_reg_train, y_cls_train, feature_names, _ = prepare_data("train")
    X_val, y_reg_val, y_cls_val, _, _ = prepare_data("val", feature_cols=feature_names)
    X_test, y_reg_test, y_cls_test, _, df_test = prepare_data("test", feature_cols=feature_names)

    if X_test is None:
        print("ERROR: No test data.")
        return None

    # Reuse the scalers fitted during training. Evaluating unscaled features
    # against models trained on scaled ones would silently report nonsense.
    from .data_assembly import (
        load_feature_scaler, apply_feature_scaler,
        load_target_scaler, apply_target_scaler,
    )
    scaler = load_feature_scaler()
    if scaler is not None:
        X_train = apply_feature_scaler(X_train, scaler)
        X_val = apply_feature_scaler(X_val, scaler)
        X_test = apply_feature_scaler(X_test, scaler)
    else:
        print("  WARNING: no saved feature scaler; evaluating on raw features.")

    # Targets must be in the same (standardised) space the models were trained
    # in, otherwise every RMSE/R2 below is computed against a different scale.
    target_scaler = load_target_scaler()
    if target_scaler is not None:
        y_reg_train = apply_target_scaler(y_reg_train, target_scaler)
        y_reg_val = apply_target_scaler(y_reg_val, target_scaler)
        y_reg_test = apply_target_scaler(y_reg_test, target_scaler)
    else:
        print("  WARNING: no saved target scaler; metrics may be on a different scale.")

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

    n_tickers = len(TICKERS)
    input_size = X_tr_seq.shape[2]

    mlflow.set_experiment("CAF_Stock_Prediction")
    run = mlflow.start_run(run_name="evaluate")
    results = {}
    # Collected per-model failures so evaluate mode can fail loudly at the end
    # instead of exiting successfully with an empty results dict.
    eval_failures = []

    try:
        mlflow.log_param("mode", "evaluate")
        mlflow.log_param("n_samples_test", X_test.shape[0])

        # Determine which time series models to evaluate
        ts_models_to_evaluate = []
        if model_filter:
            ts_model_names = ["lstm", "gru", "transformer", "bilstm", "tcn"]
            if model_filter in ts_model_names:
                ts_models_to_evaluate = [model_filter]
        elif ts_best_name:
            ts_models_to_evaluate = [ts_best_name]
        else:
            ts_models_to_evaluate = ["lstm", "gru", "transformer", "bilstm", "tcn"]

        for model_name in ts_models_to_evaluate:
            try:
                model_path = os.path.join(MODELS_DIR, f"timeseries_{model_name}.pt")
                if not os.path.exists(model_path):
                    print(f"  Skipping {model_name}: no saved weights")
                    continue

                from ..timeseries import lstm, gru, transformer, bilstm, tcn
                from ..timeseries.train import load_model, validate, get_device

                model_modules = {
                    "lstm": lstm, "gru": gru, "transformer": transformer,
                    "bilstm": bilstm, "tcn": tcn,
                }

                mod = model_modules[model_name]
                # n_tickers goes through default_kwargs: load_model() takes no
                # **kwargs, so passing it directly raised TypeError for every
                # model and the error was swallowed by the except below.
                model = load_model(
                    mod.create_model, f"timeseries_{model_name}",
                    default_kwargs={"n_tickers": n_tickers},
                )

                device = get_device()
                import torch
                from torch.utils.data import DataLoader, TensorDataset
                ds_test = TensorDataset(torch.FloatTensor(X_te_seq), torch.FloatTensor(y_reg_te), torch.FloatTensor(y_cls_te))
                test_loader = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)

                val_loss, (reg_pred, cls_pred, reg_true, cls_true) = validate(model, test_loader, device)

                from ..tabular.regress import evaluate_regression
                from ..tabular.classify import evaluate_classification

                reg_metrics = evaluate_regression(reg_true, reg_pred)
                cls_metrics = evaluate_classification(cls_true, (cls_pred > 0.5).astype(int), cls_pred)

                # Store the raw prediction arrays, not just the metrics.
                # run_detailed_evaluation reads reg_pred/cls_pred/cls_prob from
                # here; without them every --eval-* flag printed "No predictions
                # for ..." and no plots were ever produced. `model` is kept for
                # feature importance and stripped again before pickling.
                results[f"ts_{model_name}"] = {
                    "regression": reg_metrics,
                    "classification": cls_metrics,
                    "val_loss": val_loss,
                    "reg_pred": reg_pred,
                    "cls_pred": (cls_pred > 0.5).astype(int),
                    "cls_prob": cls_pred,
                    "model": model,
                }

                print(f"\n  {model_name.upper()} (test):")
                print(f"    Regression  — RMSE: {reg_metrics['rmse']:.6f} | MAE: {reg_metrics['mae']:.6f} | R2: {reg_metrics['r2']:.4f}")
                print(f"    Classification — F1: {cls_metrics.get('f1', 0):.4f} | Acc: {cls_metrics.get('accuracy', 0):.4f}")

                mlflow.log_metric(f"eval_ts_{model_name}_rmse", reg_metrics["rmse"])
                mlflow.log_metric(f"eval_ts_{model_name}_f1", cls_metrics.get("f1", 0))
                mlflow.log_metric(f"eval_ts_{model_name}_val_loss", val_loss)

            except Exception as e:
                # Record the failure instead of only printing it. This handler
                # previously hid TypeErrors from load_model and shape mismatches,
                # letting `--mode evaluate` (and CI, which runs exactly that)
                # exit 0 while evaluating nothing at all.
                import traceback
                print(f"  ERROR evaluating {model_name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                eval_failures.append(f"ts_{model_name}: {type(e).__name__}: {e}")

        # Determine which tabular models to evaluate
        tab_models_to_evaluate = []
        if model_filter:
            tab_model_names = ["xgboost", "random_forest", "lightgbm", "mlp", "ridge", "logistic"]
            if model_filter in tab_model_names:
                tab_models_to_evaluate = [model_filter]
        else:
            if tab_reg_best_name:
                tab_models_to_evaluate.append(tab_reg_best_name)
            if tab_cls_best_name and tab_cls_best_name != tab_reg_best_name:
                tab_models_to_evaluate.append(tab_cls_best_name)
            if not tab_models_to_evaluate:
                tab_models_to_evaluate = ["xgboost", "random_forest", "lightgbm", "mlp", "ridge", "logistic"]

        for model_name in tab_models_to_evaluate:
            try:
                pkl_path = os.path.join(MODELS_DIR, f"tabular_{model_name}_reg.pkl")
                # The MLP is saved by torch as `tabular_mlp_reg.pt`. This check
                # previously looked for a `tabular_mlp_reg/` DIRECTORY, which is
                # never created, so the MLP was always reported as "no saved
                # weights" even when it had just been selected as the best model.
                mlp_path = os.path.join(MODELS_DIR, f"tabular_{model_name}_reg.pt")
                if not os.path.exists(pkl_path) and not os.path.exists(mlp_path):
                    print(f"  Skipping {model_name}: no saved weights")
                    continue

                from ..tabular import pipe as tab_pipe
                from ..tabular.regress import evaluate_regression
                from ..tabular.classify import evaluate_classification

                # Load and evaluate regression
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        model = pickle.load(f)
                    if model_name == "lightgbm":
                        from ..tabular.lightgbm_model import predict_regressor as lgb_predict
                        pred_reg = lgb_predict(model, X_te_flat)
                    elif model_name in ("xgboost", "random_forest", "ridge"):
                        pred_reg = model.predict(X_te_flat)
                    else:
                        continue
                elif os.path.exists(mlp_path):
                    from ..tabular.mlp import predict_regressor, load_model as load_mlp_model
                    model = load_mlp_model(f"tabular_{model_name}_reg")
                    pred_reg = predict_regressor(model, X_te_flat)
                else:
                    continue

                # Compare against the sequence-aligned targets. X_te_flat comes
                # from X_te_seq, which drops the first SEQ_LEN rows, so it has
                # 292 rows while y_reg_test still has 322. Using y_reg_test here
                # made every evaluate_regression call raise a shape error that
                # the enclosing handler then swallowed.
                reg_metrics = evaluate_regression(y_reg_te, pred_reg)

                # Load and evaluate classification
                cls_pkl_path = os.path.join(MODELS_DIR, f"tabular_{model_name}_cls.pkl")
                # Same `.pt` vs directory mismatch as the regression path above.
                cls_mlp_path = os.path.join(MODELS_DIR, f"tabular_{model_name}_cls.pt")
                if os.path.exists(cls_pkl_path):
                    with open(cls_pkl_path, "rb") as f:
                        cls_model = pickle.load(f)
                    if model_name == "lightgbm":
                        from ..tabular.lightgbm_model import predict_classifier as lgb_predict
                        pred_cls, prob_cls = lgb_predict(cls_model, X_te_flat)
                    elif model_name in ("xgboost", "random_forest"):
                        pred_cls = cls_model.predict(X_te_flat)
                        prob_cls = cls_model.predict_proba(X_te_flat)
                    elif model_name == "logistic":
                        # Shaped from y_cls_te (sequence-aligned), not y_cls_test,
                        # so the per-ticker columns line up with X_te_flat rows.
                        pred_cls = np.zeros_like(y_cls_te, dtype=float)
                        prob_cls = np.zeros_like(y_cls_te, dtype=float)
                        for t in range(n_tickers):
                            pred_cls[:, t] = cls_model[t].predict(X_te_flat)
                            prob_cls[:, t] = cls_model[t].predict_proba(X_te_flat)[:, 1]
                    else:
                        continue
                elif os.path.exists(cls_mlp_path):
                    from ..tabular.mlp import predict_classifier, load_model as load_mlp_model
                    cls_model = load_mlp_model(f"tabular_{model_name}_cls")
                    pred_cls, prob_cls = predict_classifier(cls_model, X_te_flat)
                else:
                    pred_cls = None
                    prob_cls = None

                cls_metrics = evaluate_classification(y_cls_te, pred_cls, prob_cls) if pred_cls is not None else {}

                # Same reason as the time-series branch: the detailed evaluation
                # stage needs the arrays, not only the summary metrics.
                results[f"tab_{model_name}"] = {
                    "regression": reg_metrics,
                    "classification": cls_metrics,
                    "reg_pred": pred_reg,
                    "cls_pred": pred_cls,
                    "cls_prob": prob_cls,
                    "model": model,
                }

                print(f"\n  {model_name.upper()} (test):")
                print(f"    Regression  — RMSE: {reg_metrics['rmse']:.6f} | MAE: {reg_metrics['mae']:.6f} | R2: {reg_metrics['r2']:.4f}")
                if cls_metrics:
                    print(f"    Classification — F1: {cls_metrics.get('f1', 0):.4f} | Acc: {cls_metrics.get('accuracy', 0):.4f}")

                mlflow.log_metric(f"eval_tab_{model_name}_rmse", reg_metrics["rmse"])
                if cls_metrics:
                    mlflow.log_metric(f"eval_tab_{model_name}_f1", cls_metrics.get("f1", 0))

            except Exception as e:
                import traceback
                print(f"  ERROR evaluating {model_name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                eval_failures.append(f"tabular_{model_name}: {type(e).__name__}: {e}")

        # === Naive baselines ===
        #
        # Without a reference point, an R2 or F1 number says nothing about skill.
        # Two trivial predictors are scored on exactly the same targets:
        #   - zero return  (i.e. "tomorrow equals today"), the standard
        #     random-walk benchmark for return prediction
        #   - always-up direction, the majority class in a rising market
        # A model that cannot beat these has learned nothing useful.
        try:
            from ..tabular.regress import evaluate_regression as _eval_reg
            from ..tabular.classify import evaluate_classification as _eval_cls

            zero_pred = np.zeros_like(y_reg_te)
            base_reg = _eval_reg(y_reg_te, zero_pred)

            up_pred = np.ones_like(y_cls_te)
            base_cls = _eval_cls(y_cls_te, up_pred, up_pred.astype(float))

            results["_baseline"] = {"regression": base_reg, "classification": base_cls}

            print("\n  BASELINE (no-change return / always-up direction):")
            print(f"    Regression  — RMSE: {base_reg['rmse']:.6f} | MAE: {base_reg['mae']:.6f} | R2: {base_reg['r2']:.4f}")
            print(f"    Classification — F1: {base_cls.get('f1', 0):.4f} | Acc: {base_cls.get('accuracy', 0):.4f}")
            mlflow.log_metric("baseline_rmse", base_reg["rmse"])
            mlflow.log_metric("baseline_f1", base_cls.get("f1", 0))

            beaten = [k for k, v in results.items()
                      if not k.startswith("_")
                      and v.get("regression", {}).get("rmse", float("inf")) < base_reg["rmse"]]
            print(f"    Models beating the baseline RMSE: {beaten if beaten else 'NONE'}")
        except Exception as e:
            print(f"  Baseline comparison skipped: {type(e).__name__}: {e}")

        # Save evaluation results.
        #
        # Live model objects are dropped from the pickle: a torch module or a
        # booster is large and not reliably loadable across environments, and the
        # weights already live in their own checkpoints. Metrics and prediction
        # arrays are what this file is for.
        eval_path = os.path.join(MODELS_DIR, "evaluation_results.pkl")
        picklable = {
            k: {kk: vv for kk, vv in v.items() if kk != "model"} if isinstance(v, dict) else v
            for k, v in results.items()
        }
        with open(eval_path, "wb") as f:
            pickle.dump(picklable, f)
        print(f"\nEvaluation results saved to {eval_path}")

        # Run detailed evaluation if any flags enabled
        if eval_flags and any(eval_flags.values()):
            print("\n" + "=" * 60)
            print("DETAILED EVALUATION")
            print("=" * 60)
            try:
                from .evaluate import run_detailed_evaluation
                # Pass the SEQUENCE-ALIGNED targets: every stored prediction was
                # produced from X_te_seq, so comparing against y_reg_test would
                # reintroduce the row-count mismatch this evaluation depends on.
                run_detailed_evaluation(
                    results=results,
                    X_test=X_test,
                    y_reg_test=y_reg_te,
                    y_cls_test=y_cls_te,
                    X_te_seq=X_te_seq,
                    X_te_flat=X_te_flat,
                    df_test=df_test,
                    feature_names=feature_names,
                    eval_flags=eval_flags,
                )
            except Exception as e:
                print(f"  Detailed evaluation error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        if eval_failures:
            print("EVALUATION FINISHED WITH ERRORS")
            print("=" * 60)
            for msg in eval_failures:
                print(f"  - {msg}")
            mlflow.log_param("eval_failures", len(eval_failures))
            if not results:
                # Nothing evaluated at all. Raising is essential: CI runs
                # `--mode evaluate`, so a silent empty result set made the
                # pipeline look healthy while every model failed to load.
                raise RuntimeError(
                    "Evaluation produced no results; all models failed:\n  "
                    + "\n  ".join(eval_failures)
                )
        else:
            print("EVALUATION COMPLETE")
            print("=" * 60)

        return results
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

    reg_preds = []

    # Time series best
    ts_model = ts_results[ts_best_name]["model"]
    from ..timeseries.train import predict as ts_predict
    ts_loader = _make_loader(X_v_seq)
    ts_reg, ts_cls = ts_predict(ts_model, ts_loader)
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
    weights = [0.6, 0.4]
    ensemble_reg = sum(w * p for w, (_, p) in zip(weights, reg_preds))

    cls_preds = []
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
        tab_cls_predict = None
    else:
        tab_cls_predict = None

    if tab_cls_predict is not None:
        tab_cls_pred, tab_cls_prob = tab_cls_predict(tab_cls_model, X_v_flat)
    cls_preds.append(("tab", tab_cls_pred))

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

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save ALL time series models.
    #
    # input_size and the constructor kwargs come from the config recorded when
    # the model was built. The previous code used model.reg_head.in_features,
    # which is the shared-layer width (128), not the per-timestep feature count
    # (~1324), and saved no kwargs at all — so hidden_size/num_layers were lost.
    # Every checkpoint written that way rebuilt at the wrong shape and failed to
    # load, which is why --mode evaluate silently produced nothing.
    if ts_results:
        from ..timeseries.pipe import get_build_config
        for name, res in ts_results.items():
            model = res["model"]
            input_size, kwargs = get_build_config(model)
            if input_size is None:
                raise ValueError(
                    f"timeseries_{name}: build config missing; cannot save a "
                    "loadable checkpoint."
                )
            save_ts(model, f"timeseries_{name}", input_size, **kwargs)

    # Save tabular models
    if tab_reg_results:
        for name, res in tab_reg_results.items():
            if name.startswith("_"):
                continue
            model = res["model"]
            if name in ("xgboost", "lightgbm", "random_forest"):
                with open(os.path.join(MODELS_DIR, f"tabular_{name}_reg.pkl"), "wb") as f:
                    pickle.dump(model, f)
            elif name == "mlp":
                # save_mlp reads model.build_config; no shape reverse-engineering.
                from ..tabular.mlp import save_model as save_mlp
                save_mlp(model, f"tabular_{name}_reg")
            elif name == "ridge":
                with open(os.path.join(MODELS_DIR, f"tabular_{name}_reg.pkl"), "wb") as f:
                    pickle.dump(model, f)

    if tab_cls_results:
        for name, res in tab_cls_results.items():
            if name.startswith("_"):
                continue
            model = res["model"]
            if name in ("xgboost", "lightgbm", "random_forest"):
                with open(os.path.join(MODELS_DIR, f"tabular_{name}_cls.pkl"), "wb") as f:
                    pickle.dump(model, f)
            elif name == "mlp":
                from ..tabular.mlp import save_model as save_mlp
                save_mlp(model, f"tabular_{name}_cls", is_classifier=True)
            elif name == "logistic":
                with open(os.path.join(MODELS_DIR, f"tabular_{name}_cls.pkl"), "wb") as f:
                    pickle.dump(model, f)

    # Save feature names
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    # Save ensemble
    if ensemble:
        with open(os.path.join(MODELS_DIR, "ensemble.pkl"), "wb") as f:
            pickle.dump(ensemble, f)

    print(f"  All models saved to {MODELS_DIR}")


def save_best_model_selection(ts_best, ts_results, tab_reg_best, tab_reg_results,
                              tab_cls_best, tab_cls_results, feature_names):
    """Save best model selection to JSON for evaluate mode."""
    import json

    selection = {
        "timeseries": {},
        "tabular_regression": {},
        "tabular_classification": {},
        "features": {
            "n_features": len(feature_names),
            "feature_names_file": "feature_names.pkl",
        }
    }

    if ts_best and ts_results and ts_best in ts_results:
        selection["timeseries"] = {
            "best_model": ts_best,
            "val_loss": float(ts_results[ts_best].get("val_loss", 0)),
            "weights_file": f"timeseries_{ts_best}.pt",
        }

    if tab_reg_best and tab_reg_results and tab_reg_best in tab_reg_results:
        reg_metrics = tab_reg_results[tab_reg_best].get("overall", {})
        selection["tabular_regression"] = {
            "best_model": tab_reg_best,
            "rmse": float(reg_metrics.get("rmse", 0)),
            "weights_file": f"tabular_{tab_reg_best}_reg.pkl" if tab_reg_best != "mlp" else "tabular_mlp_reg/",
        }

    if tab_cls_best and tab_cls_results and tab_cls_best in tab_cls_results:
        cls_metrics = tab_cls_results[tab_cls_best].get("overall", {})
        selection["tabular_classification"] = {
            "best_model": tab_cls_best,
            "f1": float(cls_metrics.get("f1", 0)),
            "weights_file": f"tabular_{tab_cls_best}_cls.pkl" if tab_cls_best != "mlp" else "tabular_mlp_cls/",
        }

    path = os.path.join(MODELS_DIR, "best_model_selection.json")
    with open(path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"  Best model selection saved to {path}")
    return selection


def load_best_model_selection():
    """Load best model selection from JSON."""
    import json
    path = os.path.join(MODELS_DIR, "best_model_selection.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
