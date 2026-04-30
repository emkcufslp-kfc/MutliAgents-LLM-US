# Data Directory
This directory is used for the "As-Of-Date" Data Engine (Phase 2).

Subdirectories will include:
- `raw/`: Raw JSON/CSV dumps from yfinance and SEC.
- `processed/`: Cleaned Parquet files.
- `cache/`: DuckDB local caching databases.
- `audit/`: Logs of missing data or lookahead warnings.
