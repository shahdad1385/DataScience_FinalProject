import os
import pandas as pd
import yfinance as yf
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

INDICATORS = {
    "VIX": "^VIX",
    "US_10Y_Treasury": "^TNX",
    "US_5Y_Treasury": "^FVX",
    "US_30Y_Treasury": "^TYX",
    "Semiconductor_ETF_SMH": "SMH",
    "Tech_ETF_XLK": "XLK",
    "Bitcoin": "BTC-USD",
    "US_Dollar_Index": "DX-Y.NYB",
    "SOX_Semiconductor_Index": "^SOX",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Gold": "GC=F",
    "Crude_Oil": "CL=F",
    "Natural_Gas": "NG=F",
    "Fed_Funds_Rate": "^IRX",
}

START_DATE = "2018-01-01"


def fetch_indicators():
    print("Fetching Market Indicators via yfinance\n")

    end_date = datetime.today().strftime('%Y-%m-%d')
    frames = []

    for name, ticker in INDICATORS.items():
        print(f"Fetching {name} ({ticker})...")
        try:
            data = yf.Ticker(ticker).history(start=START_DATE, end=end_date, interval="1d")
            if data.empty:
                print(f"  No data for {ticker}, skipping.")
                continue

            df = data.reset_index()
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df["Indicator"] = name
            df["Ticker"] = ticker

            frames.append(df)
            print(f"  {len(df)} rows")
        except Exception as e:
            print(f"  Failed: {e}")

    if frames:
        master = pd.concat(frames, ignore_index=True)
        master.sort_values(by=["Date", "Indicator"], inplace=True)
        master.reset_index(drop=True, inplace=True)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "market_indicators.csv")
        master.to_csv(output_path, index=False)

        print(f"\n{'=' * 50}")
        print(f"Market Indicators Saved!")
        print(f"Total rows: {len(master)}")
        print(f"File: {os.path.abspath(output_path)}")
        print(f"{'=' * 50}")
        print(f"\nIndicators included ({len(master['Indicator'].unique())} total):")
        for name in master["Indicator"].unique():
            count = len(master[master["Indicator"] == name])
            print(f"  - {name}: {count} rows")
    else:
        print("No indicator data was collected.")


if __name__ == "__main__":
    fetch_indicators()
