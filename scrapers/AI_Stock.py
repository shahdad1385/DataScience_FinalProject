import os
import pandas as pd
import yfinance as yf
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')


def fetch_ai_ecosystem_data():
    print("Starting AI Infrastructure Data Collection Pipeline...\n")

    ticker_dict = {
        "NVIDIA": "NVDA",
        "META": "META",
        "GOOGLE": "GOOGL",
        "TSMC": "TSM",
        "VERTIV": "VRT",
        "MODINE_COOLING": "MOD",
        "SUPER_MICRO_COOLING": "SMCI",
        "AMD": "AMD",
        "INTEL": "INTC",
        "BROADCOM": "AVGO",
        "QUALCOMM": "QCOM",
        "MARVELL": "MRVL",
        "APPLIED_MATERIALS": "AMAT",
        "MICRON": "MU",
        "LAM_RESEARCH": "LRCX",
        "KLA": "KLAC",
        "ASML": "ASML",
        "ARM": "ARM",
        "NXP": "NXPI",
        "ON_SEMI": "ON",
    }

    start_date = "2018-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    frames = []

    for company_name, ticker in ticker_dict.items():
        print(f"Fetching historical data for {company_name} ({ticker})...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")

            if df.empty:
                print(f"  No data found for {ticker}. Skipping.")
                continue

            df = df.reset_index()
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Company'] = company_name
            df['Ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            frames.append(df)
            print(f"  Collected {len(df)} rows for {company_name}.")

        except Exception as e:
            print(f"  Failed to fetch data for {ticker}. Error: {e}")

    if frames:
        master_df = pd.concat(frames, ignore_index=True)

        output_path = os.path.join(OUTPUT_DIR, "ai_infrastructure_stock_data.csv")

        master_df = master_df.sort_values(by=['Date', 'Company']).reset_index(drop=True)

        master_df.to_csv(output_path, index=False)

        print("\n" + "=" * 50)
        print("DATA COLLECTION SUCCESSFUL!")
        print(f"Total Dataset Rows: {len(master_df)}")
        print(f"Tickers: {master_df['Ticker'].nunique()}")
        print(f"Saved file as: {os.path.abspath(output_path)}")
        print("=" * 50)

        print("\nDataset Sample Preview:")
        print(master_df.head(10))
    else:
        print("Pipeline completed, but no data was collected.")


if __name__ == "__main__":
    fetch_ai_ecosystem_data()
