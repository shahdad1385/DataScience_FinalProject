"""
Detailed Evaluation Module — extended metrics, confusion matrices, trading simulation.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import mlflow

from .data_assembly import TICKERS, SEQ_LEN, REF_CLOSE_SUFFIX

# Must match every other module (scripts/models). This file previously applied
# dirname three times instead of two, resolving to CAF/models, so evaluation
# artifacts were written to a directory nothing else reads.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EVAL_RESULTS_DIR = os.path.join(MODELS_DIR, "eval_results")
EVAL_PLOTS_DIR = os.path.join(MODELS_DIR, "eval_plots")
CONFUSION_DIR = os.path.join(EVAL_RESULTS_DIR, "confusion")
FEATURE_IMP_DIR = os.path.join(EVAL_RESULTS_DIR, "feature_importance")
PREDICTIONS_DIR = os.path.join(EVAL_RESULTS_DIR, "predictions")

for d in [EVAL_RESULTS_DIR, EVAL_PLOTS_DIR, CONFUSION_DIR, FEATURE_IMP_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)


def run_detailed_evaluation(results, X_test, y_reg_test, y_cls_test,
                            X_te_seq, X_te_flat, df_test, feature_names, eval_flags):
    """Run all detailed evaluations based on flags."""
    n_tickers = len(TICKERS)
    reg_output_names = [f"{t}_{o}" for t in TICKERS for o in ["open", "high", "low", "close"]]
    cls_output_names = [f"{t}_direction" for t in TICKERS]

    for model_key, model_res in results.items():
        if model_key.startswith("_"):
            continue

        print(f"\n  --- Detailed evaluation for {model_key} ---")

        reg_pred = None
        cls_pred = None
        cls_prob = None

        if "regression" in model_res and "classification" in model_res:
            reg_metrics = model_res["regression"]
            cls_metrics = model_res["classification"]

        # Read the prediction arrays unconditionally. These were previously only
        # read inside `if "model" in model_res`, but evaluate mode rebuilds models
        # locally and never stores a "model" key, so reg_pred/cls_pred stayed None
        # and every stage reported "No predictions for ..." and saved no plots.
        reg_pred = model_res.get("reg_pred")
        cls_pred = model_res.get("cls_pred")
        cls_prob = model_res.get("cls_prob")

        if eval_flags.get("detailed"):
            evaluate_detailed(
                model_key, y_reg_test, y_cls_test,
                reg_pred, cls_pred, cls_prob,
                reg_output_names, cls_output_names
            )

        if eval_flags.get("confusion"):
            save_confusion_matrices(
                model_key, y_cls_test, cls_pred, TICKERS
            )

        if eval_flags.get("trading"):
            evaluate_trading(
                model_key, y_cls_test, cls_pred, df_test, TICKERS
            )

        if eval_flags.get("feature_imp") and "model" in model_res:
            extract_feature_importance(
                model_key, model_res["model"], feature_names, X_te_flat, y_reg_test, y_cls_test
            )

        if eval_flags.get("statistical"):
            run_statistical_tests(
                model_key, y_reg_test, y_cls_test, reg_pred, cls_pred
            )

        if eval_flags.get("calibration"):
            plot_calibration_curves(
                model_key, y_cls_test, cls_prob, TICKERS
            )

        if eval_flags.get("save_preds"):
            save_predictions(
                model_key, y_reg_test, y_cls_test,
                reg_pred, cls_pred, cls_prob,
                df_test, reg_output_names, cls_output_names
            )

        if eval_flags.get("ensemble") and "ensemble" in model_res:
            evaluate_ensemble(model_res["ensemble"], y_reg_test, y_cls_test)


def evaluate_detailed(model_key, y_reg_true, y_cls_true,
                      reg_pred, cls_pred, cls_prob,
                      reg_output_names, cls_output_names):
    """Per-ticker & per-output metrics breakdown."""
    from ..tabular.regress import evaluate_regression, evaluate_per_output
    from ..tabular.classify import evaluate_classification, evaluate_per_ticker

    print(f"    Computing detailed metrics...")

    reg_detailed = {}
    if reg_pred is not None:
        reg_detailed = evaluate_per_output(y_reg_true, reg_pred, reg_output_names)

    cls_detailed = {}
    if cls_pred is not None and cls_prob is not None:
        cls_detailed = evaluate_per_ticker(y_cls_true, cls_pred, cls_prob, TICKERS)

    all_metrics = []
    for out_name, metrics in reg_detailed.items():
        for m_name, m_val in metrics.items():
            all_metrics.append({
                "model": model_key,
                "type": "regression",
                "output": out_name,
                "metric": m_name,
                "value": m_val
            })
            mlflow.log_metric(f"eval_{model_key}_reg_{out_name}_{m_name}", m_val)

    for ticker_name, metrics in cls_detailed.items():
        for m_name, m_val in metrics.items():
            all_metrics.append({
                "model": model_key,
                "type": "classification",
                "output": ticker_name,
                "metric": m_name,
                "value": m_val
            })
            mlflow.log_metric(f"eval_{model_key}_cls_{ticker_name}_{m_name}", m_val)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        out_path = os.path.join(EVAL_RESULTS_DIR, f"detailed_metrics_{model_key}.csv")
        df_metrics.to_csv(out_path, index=False)
        mlflow.log_artifact(out_path)
        print(f"    Saved detailed metrics to {out_path}")


def save_confusion_matrices(model_key, y_true, y_pred, ticker_names):
    """Save confusion matrices as PNG + CSV per ticker."""
    if y_pred is None:
        print(f"    No predictions for confusion matrix")
        return

    # Single-target classifiers can emit 1-D arrays; coerce to (n, n_tickers)
    # so the [:, i] indexing below is safe.
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    for i, ticker in enumerate(ticker_names):
        y_t = y_true[:, i]
        y_p = (y_pred[:, i] > 0.5).astype(int)

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{model_key} - {ticker} Confusion Matrix")

        disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
        disp.plot(ax=axes[0], cmap="Blues", values_format="d")
        axes[0].set_title("Absolute Counts")

        disp_norm = ConfusionMatrixDisplay(cm_norm, display_labels=["Down", "Up"])
        disp_norm.plot(ax=axes[1], cmap="Blues", values_format=".2%")
        axes[1].set_title("Normalized (%)")

        plt.tight_layout()
        png_path = os.path.join(CONFUSION_DIR, f"{model_key}_ticker_{ticker}.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()

        df_cm = pd.DataFrame(cm, index=["True_Down", "True_Up"], columns=["Pred_Down", "Pred_Up"])
        csv_path = os.path.join(CONFUSION_DIR, f"{model_key}_ticker_{ticker}.csv")
        df_cm.to_csv(csv_path)

        mlflow.log_artifact(png_path)
        mlflow.log_artifact(csv_path)

    print(f"    Saved {len(ticker_names)} confusion matrices to {CONFUSION_DIR}")


def _actual_next_day_returns(df_test, ticker_names):
    """Realised next-day return per ticker, taken from the joint frame.

    Each ticker's today-close lives in {TICKER}_refclose, so the realised
    next-day return is refclose.shift(-1) / refclose - 1. The final row has no
    next day and is returned as NaN, then filtered by the caller.
    """
    if df_test is None or "date" not in getattr(df_test, "columns", []):
        return {}

    out = {}
    ordered = df_test.sort_values("date")
    for ticker in ticker_names:
        col = f"{ticker}_{REF_CLOSE_SUFFIX}"
        if col not in ordered.columns:
            continue
        ref = pd.to_numeric(ordered[col], errors="coerce")
        ref = ref.where(ref > 0)
        out[ticker] = (ref.shift(-1) / ref - 1.0).values
    return out


def evaluate_trading(model_key, y_true, y_pred, df_test, ticker_names):
    """Trading simulation: long-only on predicted direction."""
    if y_pred is None:
        print(f"    No predictions for trading evaluation")
        return

    # df_test is the JOINT frame: one row per date, with each ticker's reference
    # close in its own {TICKER}_refclose column. The previous code indexed
    # ["date", "ticker", "close"], which only exists in the pre-pivot long
    # format, so trading evaluation raised KeyError and was silently skipped.
    actual_by_ticker = _actual_next_day_returns(df_test, ticker_names)
    if not actual_by_ticker:
        print("    No reference prices in df_test for trading evaluation")
        return

    returns_list = []
    for i, ticker in enumerate(ticker_names):
        actual_returns = actual_by_ticker.get(ticker)
        if actual_returns is None or len(actual_returns) < 2:
            continue

        # Align predictions with realised returns and drop the trailing row,
        # whose next-day return is unknown.
        n = min(len(actual_returns), len(y_pred))
        actual_returns = np.asarray(actual_returns[:n], dtype=float)
        preds = np.asarray(y_pred[:n, i], dtype=float)
        keep = np.isfinite(actual_returns) & np.isfinite(preds)
        actual_returns = actual_returns[keep]
        preds = preds[keep]
        if len(actual_returns) < 2:
            continue

        # Works for both probabilities and hard 0/1 labels.
        signals = (preds > 0.5).astype(int)
        strategy_returns = signals * actual_returns

        returns_list.append({
            "ticker": ticker,
            "strategy_returns": strategy_returns,
            "actual_returns": actual_returns,
            "signals": signals,
        })

    if not returns_list:
        print(f"    Insufficient data for trading simulation")
        return

    n_days = min(len(r["strategy_returns"]) for r in returns_list)
    portfolio_returns = np.zeros(n_days)
    for r in returns_list:
        portfolio_returns += r["strategy_returns"][:n_days] / len(returns_list)

    risk_free_daily = 0.0
    try:
        from scripts.db import get_engine
        engine = get_engine()
        rf_df = pd.read_sql("SELECT date, close FROM market_indicators WHERE indicator='US_5Y_Treasury' ORDER BY date", engine, parse_dates=["date"])
        if not rf_df.empty:
            rf_daily = (rf_df.set_index("date")["close"] / 100 / 252).reindex(pd.date_range(start=rf_df["date"].min(), end=rf_df["date"].max(), freq="D")).ffill()
            risk_free_daily = rf_daily.iloc[-n_days:].values.mean()
    except Exception:
        pass

    total_return = np.prod(1 + portfolio_returns) - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = (ann_return - risk_free_daily * 252) / (ann_vol + 1e-8)

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else ann_vol
    sortino = (ann_return - risk_free_daily * 252) / (downside_vol + 1e-8)

    cum_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_return / (abs(max_dd) + 1e-8)

    all_signals = np.concatenate([r["signals"][:n_days] for r in returns_list])
    all_actual = np.concatenate([(r["actual_returns"][:n_days] > 0).astype(int) for r in returns_list])
    hit_rate = (all_signals == all_actual).mean()

    wins = portfolio_returns[portfolio_returns > 0]
    losses = portfolio_returns[portfolio_returns < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    bh_returns = np.mean([r["actual_returns"][:n_days] for r in returns_list], axis=0)
    up_mask = bh_returns > 0
    down_mask = bh_returns < 0
    up_capture = portfolio_returns[up_mask].mean() / bh_returns[up_mask].mean() if up_mask.any() and bh_returns[up_mask].mean() != 0 else 0
    down_capture = portfolio_returns[down_mask].mean() / bh_returns[down_mask].mean() if down_mask.any() and bh_returns[down_mask].mean() != 0 else 0

    metrics = {
        "model": model_key,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "n_days": n_days,
        "n_tickers": len(returns_list),
    }

    df_metrics = pd.DataFrame([metrics])
    out_path = os.path.join(EVAL_RESULTS_DIR, f"trading_metrics_{model_key}.csv")
    df_metrics.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path)

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"eval_{model_key}_trading_{k}", v)

    print(f"    Trading metrics: Sharpe={sharpe:.3f}, Sortino={sortino:.3f}, MaxDD={max_dd:.3%}, HitRate={hit_rate:.3%}")


def extract_feature_importance(model_key, model, feature_names, X_test, y_reg_test, y_cls_test):
    """Feature importance for tree models; permutation for others."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            source = "builtin"
        else:
            from ..tabular.regress import evaluate_regression
            result = permutation_importance(
                model, X_test, y_reg_test[:, 0], n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
            source = "permutation"

        df_imp = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances,
            "source": source
        }).sort_values("importance", ascending=False)

        out_path = os.path.join(FEATURE_IMP_DIR, f"{model_key}_importance.csv")
        df_imp.to_csv(out_path, index=False)
        mlflow.log_artifact(out_path)

        top20 = df_imp.head(20)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top20)), top20["importance"][::-1])
        plt.yticks(range(len(top20)), top20["feature"][::-1])
        plt.xlabel("Importance")
        plt.title(f"{model_key} - Top 20 Feature Importance ({source})")
        plt.tight_layout()
        png_path = os.path.join(EVAL_PLOTS_DIR, f"{model_key}_feature_importance.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(png_path)

        print(f"    Feature importance saved ({source})")
    except Exception as e:
        print(f"    Feature importance error: {e}")


def run_statistical_tests(model_key, y_reg_true, y_cls_true, reg_pred, cls_pred):
    """Diebold-Mariano test vs naive baseline; permutation importance."""
    try:
        from statsmodels.tsa.stattools import acf
        from scipy.stats import norm

        if reg_pred is not None:
            naive_pred = np.roll(y_reg_true, 1, axis=0)
            naive_pred[0] = y_reg_true[0]

            for i in range(y_reg_true.shape[1]):
                e1 = (y_reg_true[:, i] - reg_pred[:, i]) ** 2
                e2 = (y_reg_true[:, i] - naive_pred[:, i]) ** 2
                d = e1 - e2
                d_mean = d.mean()
                d_var = d.var(ddof=1)

                if d_var > 0:
                    n = len(d)
                    dm_stat = d_mean / np.sqrt(d_var / n)
                    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

                    mlflow.log_metric(f"eval_{model_key}_dm_stat_{i}", dm_stat)
                    mlflow.log_metric(f"eval_{model_key}_dm_pval_{i}", p_value)

        print(f"    Statistical tests completed")
    except ImportError:
        print(f"    statsmodels not available, skipping Diebold-Mariano")
    except Exception as e:
        print(f"    Statistical tests error: {e}")


def plot_calibration_curves(model_key, y_true, y_prob, ticker_names):
    """Reliability diagrams for classifiers."""
    if y_prob is None:
        print(f"    No probabilities for calibration")
        return

    for i, ticker in enumerate(ticker_names):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        if len(np.unique(y_t)) < 2:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        from sklearn.calibration import CalibrationDisplay
        CalibrationDisplay.from_predictions(y_t, y_p, n_bins=10, ax=ax)
        ax.set_title(f"{model_key} - {ticker} Calibration")
        plt.tight_layout()
        png_path = os.path.join(EVAL_PLOTS_DIR, f"{model_key}_calibration_{ticker}.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(png_path)

    print(f"    Calibration curves saved for {len(ticker_names)} tickers")


def save_predictions(model_key, y_reg_true, y_cls_true,
                     reg_pred, cls_pred, cls_prob,
                     df_test, reg_output_names, cls_output_names):
    """Save raw predictions + probabilities to CSV.

    df_test is the JOINT frame: one row per date, carrying every ticker's columns.
    There is no single "ticker" column to read — indexing df_test["ticker"] here
    raised KeyError and the CSV was never written. Each row therefore covers all
    tickers, with the ticker encoded in the per-output column names.
    """
    n_rows = min(
        len(df_test) if df_test is not None else 0,
        len(y_reg_true) if y_reg_true is not None else 0,
    ) or (len(y_reg_true) if y_reg_true is not None else 0)

    if df_test is not None and "date" in df_test.columns:
        dates = pd.to_datetime(df_test.sort_values("date")["date"]).values[:n_rows]
    else:
        dates = np.arange(n_rows)

    pred_rows = []
    for i in range(n_rows):
        row = {"date": dates[i] if i < len(dates) else None}
        for j, name in enumerate(reg_output_names):
            row[f"true_{name}"] = y_reg_true[i, j] if i < len(y_reg_true) else np.nan
            row[f"pred_{name}"] = reg_pred[i, j] if reg_pred is not None and i < len(reg_pred) else np.nan
        for j, name in enumerate(cls_output_names):
            row[f"true_{name}"] = y_cls_true[i, j] if i < len(y_cls_true) else np.nan
            row[f"pred_{name}"] = cls_pred[i, j] if cls_pred is not None and i < len(cls_pred) else np.nan
            row[f"prob_{name}"] = cls_prob[i, j] if cls_prob is not None and i < len(cls_prob) else np.nan
        pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)
    out_path = os.path.join(PREDICTIONS_DIR, f"{model_key}_predictions.csv")
    df_preds.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path)
    print(f"    Predictions saved to {out_path}")


def evaluate_ensemble(ensemble, y_reg_true, y_cls_true):
    """Evaluate ensemble predictions."""
    if "regression" in ensemble and "classification" in ensemble:
        from ..tabular.regress import evaluate_regression
        from ..tabular.classify import evaluate_classification

        reg_metrics = evaluate_regression(y_reg_true, ensemble["regression"])
        cls_metrics = evaluate_classification(y_cls_true, ensemble["classification"])

        print(f"    Ensemble Regression: RMSE={reg_metrics['rmse']:.6f}")
        print(f"    Ensemble Classification: F1={cls_metrics.get('f1', 0):.4f}")

        mlflow.log_metric("eval_ensemble_rmse", reg_metrics["rmse"])
        mlflow.log_metric("eval_ensemble_f1", cls_metrics.get("f1", 0))