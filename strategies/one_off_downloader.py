"""
Download {SYMBOL}_{interval}_spot.csv files for all symbols in params.py into data/.

Set INTERVAL to any yfinance-supported interval. The script automatically
uses the maximum available look-back period for that interval and trims all
CSVs to the same overlapping date range so every file starts and ends on the
same timestamp.

Supported intervals and their look-back limits:
  1m  ->  7 days
  2m / 5m / 15m / 30m  ->  60 days
  1h  ->  730 days (~2 years)
  1d / 1wk / 1mo  ->  max (full history)

Usage:
    python strategies/one_off_downloader.py
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies import params as P

DATA_DIR = ROOT / "data"

# ── Set your desired interval here ──────────────────────────────────────────
INTERVAL = "1h"
# ────────────────────────────────────────────────────────────────────────────

# Maximum look-back period yfinance supports per interval
_MAX_PERIOD = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "max",
    "1wk": "max",
    "1mo": "max",
}

CRYPTO_SUFFIX = "-USD"  # yfinance crypto tickers: BTC -> BTC-USD


def _period_for(interval: str) -> str:
    period = _MAX_PERIOD.get(interval)

    if period is None:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Choose from: {list(_MAX_PERIOD)}"
        )
    return period


def _yf_ticker(symbol: str) -> str:
    if symbol in P.CRYPTO_SYMBOLS:
        return symbol + CRYPTO_SUFFIX
    return symbol


def _download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    ticker = _yf_ticker(symbol)
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index.name = "datetime"
    df = df.reset_index()
    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df[["open", "high", "low", "close", "volume", "timestamp"]]


def main():
    period = _period_for(INTERVAL)
    DATA_DIR.mkdir(exist_ok=True)
    symbols = P.STOCK_SLEEVE_SYMBOLS + P.CRYPTO_SLEEVE_SYMBOLS + [P.STOCK_BENCH]

    print(f"Interval : {INTERVAL}  (period={period})")
    print(f"Symbols  : {len(symbols)}")
    print(f"Output   : {DATA_DIR}\n")

    # ── Pass 1: download all symbols into memory ─────────────────────────
    downloaded: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for symbol in symbols:
        try:
            df = _download(symbol, INTERVAL, period)
            downloaded[symbol] = df
            print(f"  DL  {symbol:8s}  {len(df):>5} rows")
        except Exception as e:
            print(f"  ERR {symbol:8s}  {e}")
            failed.append(symbol)

    if not downloaded:
        print("\nNo data downloaded — nothing saved.")
        return

    # ── Pass 2: find overlapping window ──────────────────────────────────
    common_start = max(df["timestamp"].min() for df in downloaded.values())
    common_end   = min(df["timestamp"].max() for df in downloaded.values())

    if common_start >= common_end:
        raise RuntimeError(
            f"No overlapping date range across downloaded symbols "
            f"(start={common_start}, end={common_end}). "
            "Check that all symbols have data in the same period."
        )

    print(f"\nOverlapping window: {common_start} -> {common_end}")
    print()

    # ── Pass 3: trim to common window and save ───────────────────────────
    ok: list[str] = []

    for symbol, df in downloaded.items():
        df = df[(df["timestamp"] >= common_start) & (df["timestamp"] <= common_end)].copy()
        df["volume"] = df["volume"].replace(0, pd.NA).ffill().fillna(1)
        out = DATA_DIR / f"{symbol}_{INTERVAL}_spot.csv"
        df.to_csv(out, index=False)
        print(f"  OK  {symbol:8s}  {len(df):>5} rows  -> {out.name}")
        ok.append(symbol)

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
