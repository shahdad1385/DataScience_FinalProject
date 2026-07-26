import subprocess
import sys
import os
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.database_connection import get_engine, get_db_path

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")


def get_python():
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def drop_split_tables():
    """Drop only train_*/val_*/test_* tables. NEVER touch raw or computed
    tables (news, stock_prices, market_indicators, ...) — those hold
    llm_summary and feature-engineered columns that are expensive or
    impossible to regenerate."""
    engine = get_engine()
    dropped = 0
    with engine.connect() as conn:
        names = [r[0] for r in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()]
        for n in names:
            if n.startswith(("train_", "val_", "test_")):
                conn.execute(text(f"DROP TABLE IF EXISTS {n}"))
                dropped += 1
                print(f"  dropped {n}")
        conn.commit()
    print(f"Dropped {dropped} split table(s). Raw/news tables preserved.")


def main():
    python = get_python()
    print(f"Using Python: {python}")

    if not os.path.exists(get_db_path()):
        print(f"No DB at {get_db_path()}; load_data.py will create it.")
    else:
        print(f"Existing DB found. Preserving raw tables, dropping only splits:")
        drop_split_tables()

    scripts = [
        ("load_data.py", "Loading raw data into DB (idempotent)"),
        ("feature_engineering.py", "Engineering features"),
        ("preprocess.py", "Preprocessing (split, nulls, normalize)"),
    ]

    for script_name, desc in scripts:
        print(f"\n{'=' * 50}")
        print(f"Running: {desc}")
        print(f"{'=' * 50}")
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        subprocess.run([python, script_path], check=True)

    print(f"\n{'=' * 50}")
    print("Pipeline complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
