"""
Hyperparameter fine-tuning with Optuna.

Optimizes time series and tabular model hyperparameters on the validation set.
Each trial trains a model with sampled params and returns val metric.
"""

import os
import sys
import optuna
import numpy as np
import mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_assembly import (
    prepare_data, prepare_sequences, flatten_sequences, TICKERS, SEQ_LEN,
)


def finetune_all(trials=50, model_filter=None, verbose=False, skip_nlp=False):
    """
    Run Optuna hyperparameter search for all models.

    Args:
        trials: number of Optuna trials
        model_filter: if set, only tune this model
        verbose: print progress
        skip_nlp: skip NLP feature extraction
    """
    print("=" * 60)
    print(f"HYPERPARAMETER TUNING — {trials} trials")
    print("=" * 60)

    mlflow.set_experiment("CAF_Stock_Prediction")
    run = mlflow.start_run(run_name="finetune")

    try:
        # Load data once
        X_train, y_reg_train, y_cls_train, feature_names, _ = prepare_data("train")
        X_val, y_reg_val, y_cls_val, _, _ = prepare_data("val")

        if X_train is None:
            print("ERROR: No training data.")
            return None

        (X_tr_seq, y_reg_tr, y_cls_tr,
         X_v_seq, y_reg_v, y_cls_v,
         _, _, _) = prepare_sequences(
            X_train, y_reg_train, y_cls_train,
            X_val, y_reg_val, y_cls_val,
            X_val, y_reg_val, y_cls_val,
        )

        X_tr_flat = flatten_sequences(X_tr_seq)
        X_v_flat = flatten_sequences(X_v_seq)

        n_tickers = len(TICKERS)
        input_size = X_tr_seq.shape[2]

        mlflow.log_param("finetune_trials", trials)
        mlflow.log_param("seq_len", SEQ_LEN)
        mlflow.log_param("n_tickers", n_tickers)
        mlflow.log_param("n_features", input_size)

        best_params = {}

        # Finetune time series models
        ts_models = ["lstm", "gru", "transformer", "bilstm", "tcn"]
        if model_filter and model_filter in ts_models:
            ts_models = [model_filter]
        elif model_filter and model_filter not in ts_models:
            ts_models = []

        for model_name in ts_models:
            print(f"\n--- Tuning {model_name.upper()} ---")
            study = optuna.create_study(direction="minimize", study_name=f"ts_{model_name}")
            study.optimize(
                lambda trial: _objective_ts(trial, model_name, X_tr_seq, y_reg_tr, y_cls_tr,
                                           X_v_seq, y_reg_v, y_cls_v, input_size, n_tickers),
                n_trials=trials,
                show_progress_bar=verbose,
            )
            best_params[f"ts_{model_name}"] = study.best_params
            mlflow.log_metric(f"ts_{model_name}_best_val_loss", study.best_value)
            mlflow.log_params({f"ts_{model_name}_{k}": v for k, v in study.best_params.items()})
            print(f"  Best val_loss: {study.best_value:.6f}")
            print(f"  Best params: {study.best_params}")

        # Finetune tabular regression models
        tab_reg_models = ["xgboost", "lightgbm", "random_forest", "mlp", "ridge"]
        if model_filter and model_filter in tab_reg_models:
            tab_reg_models = [model_filter]
        elif model_filter and model_filter not in tab_reg_models:
            tab_reg_models = []

        for model_name in tab_reg_models:
            print(f"\n--- Tuning {model_name.upper()} (regression) ---")
            study = optuna.create_study(direction="minimize", study_name=f"tab_reg_{model_name}")
            study.optimize(
                lambda trial: _objective_tab_reg(trial, model_name, X_tr_flat, y_reg_tr,
                                                 X_v_flat, y_reg_v, feature_names),
                n_trials=trials,
                show_progress_bar=verbose,
            )
            best_params[f"tab_reg_{model_name}"] = study.best_params
            mlflow.log_metric(f"tab_reg_{model_name}_best_rmse", study.best_value)
            mlflow.log_params({f"tab_reg_{model_name}_{k}": v for k, v in study.best_params.items()})
            print(f"  Best RMSE: {study.best_value:.6f}")
            print(f"  Best params: {study.best_params}")

        # Finetune tabular classification models
        tab_cls_models = ["xgboost", "lightgbm", "random_forest", "mlp", "logistic"]
        if model_filter and model_filter in tab_cls_models:
            tab_cls_models = [model_filter]
        elif model_filter and model_filter not in tab_cls_models:
            tab_cls_models = []

        for model_name in tab_cls_models:
            print(f"\n--- Tuning {model_name.upper()} (classification) ---")
            study = optuna.create_study(direction="maximize", study_name=f"tab_cls_{model_name}")
            study.optimize(
                lambda trial: _objective_tab_cls(trial, model_name, X_tr_flat, y_cls_tr,
                                                 X_v_flat, y_cls_v, n_tickers),
                n_trials=trials,
                show_progress_bar=verbose,
            )
            best_params[f"tab_cls_{model_name}"] = study.best_params
            mlflow.log_metric(f"tab_cls_{model_name}_best_f1", study.best_value)
            mlflow.log_params({f"tab_cls_{model_name}_{k}": v for k, v in study.best_params.items()})
            print(f"  Best F1: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")

        # Save best params
        import pickle
        params_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(params_dir, exist_ok=True)
        with open(os.path.join(params_dir, "best_hyperparams.pkl"), "wb") as f:
            pickle.dump(best_params, f)
        print(f"\nBest hyperparameters saved to {params_dir}/best_hyperparams.pkl")

        return best_params

    finally:
        mlflow.end_run()


def _objective_ts(trial, model_name, X_tr, y_reg_tr, y_cls_tr,
                  X_v, y_reg_v, y_cls_v, input_size, n_tickers):
    """Optuna objective for time series models."""
    from ..timeseries import lstm, gru, transformer, bilstm, tcn
    from ..timeseries.train import train_model, get_device, validate
    from ..timeseries.data import create_dataloaders

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 30, 100)
    patience = trial.suggest_int("patience", 5, 15)

    # Model-specific params
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model_modules = {
        "lstm": lstm, "gru": gru, "transformer": transformer,
        "bilstm": bilstm, "tcn": tcn,
    }

    mod = model_modules[model_name]
    extra_kwargs = {}
    if model_name in ("transformer", "tcn"):
        extra_kwargs["nhead"] = trial.suggest_categorical("nhead", [2, 4, 8])

    model = mod.create_model(
        input_size, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, n_tickers=n_tickers, **extra_kwargs,
    )

    device = get_device()
    train_loader, val_loader, _ = create_dataloaders(
        X_tr, y_reg_tr, y_cls_tr,
        X_v, y_reg_v, y_cls_v,
        X_v, y_reg_v, y_cls_v,
        seq_len=SEQ_LEN, batch_size=batch_size,
    )

    model, _ = train_model(
        model, train_loader, val_loader,
        lr=lr, epochs=epochs, patience=patience,
        device=device, verbose=False,
    )

    val_loss, _ = validate(model, val_loader, device)
    return val_loss


def _objective_tab_reg(trial, model_name, X_tr, y_reg_tr, X_v, y_reg_v, feature_names):
    """Optuna objective for tabular regression models."""
    from ..tabular.regress import evaluate_regression

    if model_name == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_reg_tr, eval_set=[(X_v, y_reg_v)], verbose=False)
        pred = model.predict(X_v)

    elif model_name == "lightgbm":
        try:
            import lightgbm as lgb
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            num_leaves = trial.suggest_int("num_leaves", 20, 100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

            model = lgb.LGBMRegressor(
                n_estimators=n_estimators, num_leaves=num_leaves,
                learning_rate=learning_rate, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(X_tr, y_reg_tr, eval_set=[(X_v, y_reg_v)])
            pred = model.predict(X_v)
        except ImportError:
            return float("inf")

    elif model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_reg_tr)
        pred = model.predict(X_v)

    elif model_name == "mlp":
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        hidden_sizes = [hidden_size] * n_layers
        from ..tabular.mlp import train_regressor, predict_regressor
        model = train_regressor(
            X_tr, y_reg_tr, X_v, y_reg_v,
            hidden_sizes=hidden_sizes, lr=lr,
        )
        pred = predict_regressor(model, X_v)

    elif model_name == "ridge":
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_reg_tr)
        pred = model.predict(X_v)

    metrics = evaluate_regression(y_reg_v, pred)
    return metrics["rmse"]


def _objective_tab_cls(trial, model_name, X_tr, y_cls_tr, X_v, y_cls_v, n_tickers):
    """Optuna objective for tabular classification models."""
    from ..tabular.classify import evaluate_classification

    if model_name == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_tr_flat = le.fit_transform(y_cls_tr.ravel()).reshape(y_cls_tr.shape)
        y_v_flat = le.transform(y_cls_tr.ravel()).reshape(y_cls_v.shape) if y_cls_tr.shape[1] == y_cls_v.shape[1] else y_cls_v

        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss",
        )
        model.fit(X_tr, y_cls_tr, eval_set=[(X_v, y_cls_v)], verbose=False)
        pred = model.predict(X_v)
        prob = model.predict_proba(X_v)[:, 1] if n_tickers == 1 else model.predict_proba(X_v)

    elif model_name == "lightgbm":
        try:
            import lightgbm as lgb
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            num_leaves = trial.suggest_int("num_leaves", 20, 100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators, num_leaves=num_leaves,
                learning_rate=learning_rate, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(X_tr, y_cls_tr, eval_set=[(X_v, y_cls_v)])
            pred = model.predict(X_v)
            prob = model.predict_proba(X_v)[:, 1] if n_tickers == 1 else model.predict_proba(X_v)
        except ImportError:
            return 0.0

    elif model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_cls_tr)
        pred = model.predict(X_v)
        prob = model.predict_proba(X_v)[:, 1] if n_tickers == 1 else model.predict_proba(X_v)

    elif model_name == "mlp":
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        hidden_sizes = [hidden_size] * n_layers
        from ..tabular.mlp import train_classifier, predict_classifier
        model = train_classifier(
            X_tr, y_cls_tr, X_v, y_cls_v,
            hidden_sizes=hidden_sizes, lr=lr,
        )
        pred, prob = predict_classifier(model, X_v)

    elif model_name == "logistic":
        C = trial.suggest_float("C", 0.01, 100.0, log=True)

        from sklearn.linear_model import LogisticRegression
        from ..tabular.classify import confidence_from_probs
        preds = np.zeros_like(y_cls_v, dtype=float)
        probs = np.zeros_like(y_cls_v, dtype=float)
        for t in range(n_tickers):
            lr_model = LogisticRegression(C=C, max_iter=500, random_state=42)
            lr_model.fit(X_tr, y_cls_tr[:, t])
            preds[:, t] = lr_model.predict(X_v)
            probs[:, t] = lr_model.predict_proba(X_v)[:, 1]
        pred = preds
        prob = probs

    if n_tickers == 1 and prob.ndim > 1:
        prob = prob[:, 1]

    metrics = evaluate_classification(y_cls_v, pred, prob)
    return metrics.get("f1", 0.0)
