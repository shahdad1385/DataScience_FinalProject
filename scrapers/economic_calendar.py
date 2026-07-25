import os
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

TICKERS = [
    "NVDA", "META", "GOOGL", "TSM", "VRT", "MOD", "SMCI",
    "AMD", "INTC", "AVGO", "QCOM", "MRVL", "AMAT", "MU",
    "LRCX", "KLAC", "ASML", "ARM", "NXPI", "ON",
]


def fetch_earnings_dates():
    print("Fetching earnings dates for all tickers...\n")
    rows = []

    for ticker in TICKERS:
        print(f"  {ticker}...")
        try:
            stock = yf.Ticker(ticker)

            # Earnings dates
            try:
                cal = stock.calendar
                if cal is not None and not (isinstance(cal, dict) and "error" in cal):
                    # calendar is a dict with 'Earnings Date' and 'Earnings Call Date'
                    if isinstance(cal, dict):
                        dates = cal.get("Earnings Date", [])
                        if dates:
                            for d in dates:
                                try:
                                    date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
                                    rows.append({
                                        "date": date_str,
                                        "event_type": "earnings",
                                        "ticker": ticker,
                                        "event_name": f"{ticker} Earnings Report",
                                        "impact_level": "high",
                                    })
                                except Exception:
                                    pass
            except Exception:
                pass

            # Dividends (recent)
            try:
                divs = stock.dividends
                if divs is not None and not divs.empty:
                    recent = divs[divs.index >= "2018-01-01"]
                    for div_date in recent.index:
                        rows.append({
                            "date": div_date.strftime("%Y-%m-%d"),
                            "event_type": "dividend",
                            "ticker": ticker,
                            "event_name": f"{ticker} Dividend Payment",
                            "impact_level": "low",
                        })
            except Exception:
                pass

            # Splits
            try:
                splits = stock.splits
                if splits is not None and not splits.empty:
                    recent = splits[splits.index >= "2018-01-01"]
                    for split_date in recent.index:
                        ratio = recent[split_date]
                        rows.append({
                            "date": split_date.strftime("%Y-%m-%d"),
                            "event_type": "split",
                            "ticker": ticker,
                            "event_name": f"{ticker} Stock Split {ratio}:1",
                            "impact_level": "high",
                        })
            except Exception:
                pass

        except Exception as e:
            print(f"    Error: {e}")

    if not rows:
        print("No events collected.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(["date", "ticker"], inplace=True)
    df.drop_duplicates(subset=["date", "ticker", "event_type"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "economic_events.csv")
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"Economic Events Saved!")
    print(f"Total events: {len(df)}")
    print(f"File: {os.path.abspath(output_path)}")
    print(f"{'=' * 50}")
    print(f"\nEvent breakdown:")
    for etype, cnt in df["event_type"].value_counts().items():
        print(f"  {etype}: {cnt}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers with events: {df['ticker'].nunique()}")


if __name__ == "__main__":
    fetch_earnings_dates()
