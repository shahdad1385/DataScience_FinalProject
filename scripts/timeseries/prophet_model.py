"""
Prophet model for stock price prediction.

One model per ticker — Facebook's time series model.
Uses trend + seasonality + optional regressors (VIX, sentiment).
Not a PyTorch model — uses the prophet library.
"""

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def create_model(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                 yearly_seasonality=True, weekly_seasonality=True):
    """Create a Prophet model."""
    if not HAS_PROPHET:
        raise ImportError("prophet not installed. Run: pip install prophet")
    return Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
    )


def prepare_data(df, date_col="date", value_col="close", regressors=None):
    """
    Prepare data for Prophet. Prophet requires columns 'ds' (date) and 'y' (value).

    Args:
        df: DataFrame with date and value columns
        date_col: name of date column
        value_col: name of value column
        regressors: list of additional column names to use as regressors

    Returns:
        DataFrame with 'ds', 'y', and optional regressor columns
    """
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df[date_col]),
        "y": df[value_col].values,
    })

    if regressors:
        for reg in regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].values

    return prophet_df


def fit(model, prophet_df, regressors=None):
    """Fit Prophet model."""
    if regressors:
        for reg in regressors:
            if reg in prophet_df.columns:
                model.add_regressor(reg)
    model.fit(prophet_df)
    return model


def predict(model, periods=1, freq="D", regressors_df=None):
    """
    Make future predictions.

    Args:
        model: fitted Prophet model
        periods: number of periods to forecast
        freq: frequency ('D' for daily)
        regressors_df: DataFrame with regressor values for future periods

    Returns:
        DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)

    if regressors_df is not None:
        for col in regressors_df.columns:
            future[col] = regressors_df[col].values[:len(future)]

    forecast = model.predict(future)
    return forecast


def predict_next_day(model, last_date, regressors=None):
    """
    Predict the next day's close price.

    Args:
        model: fitted Prophet model
        last_date: the last known date
        regressors: dict of regressor values for the next day

    Returns:
        predicted close price
    """
    future = pd.DataFrame({"ds": [last_date + pd.Timedelta(days=1)]})
    if regressors:
        for k, v in regressors.items():
            future[k] = v

    forecast = model.predict(future)
    return forecast["yhat"].values[0]


def train_per_ticker(df, tickers, value_col="close", regressors=None, **kwargs):
    """
    Train one Prophet model per ticker.

    Args:
        df: DataFrame with columns [date, ticker, close, ...]
        tickers: list of ticker symbols
        value_col: column to predict
        regressors: list of regressor column names
        **kwargs: passed to Prophet constructor

    Returns:
        dict mapping ticker -> fitted Prophet model
    """
    models = {}
    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker].copy()
        if len(ticker_df) < 30:  # need minimum data
            continue

        prophet_df = prepare_data(ticker_df, value_col=value_col, regressors=regressors)
        model = create_model(**kwargs)
        model = fit(model, prophet_df, regressors=regressors)
        models[ticker] = model

    return models


def predict_per_ticker(models, tickers, periods=1, regressors_dfs=None):
    """
    Predict for each ticker.

    Args:
        models: dict mapping ticker -> fitted Prophet model
        tickers: list of ticker symbols
        periods: number of periods to forecast
        regressors_dfs: dict mapping ticker -> regressor DataFrame

    Returns:
        dict mapping ticker -> forecast DataFrame
    """
    forecasts = {}
    for ticker in tickers:
        if ticker not in models:
            continue
        reg_df = regressors_dfs.get(ticker) if regressors_dfs else None
        forecasts[ticker] = predict(models[ticker], periods=periods, regressors_df=reg_df)
    return forecasts
