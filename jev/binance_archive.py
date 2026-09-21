"""Reading Binance's public data archive (https://data.binance.vision).

Files are streamed into memory and parsed there. Nothing raw is written to disk.
"""

import io
import threading
import zipfile

import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision/data/futures/um/daily"

_thread_local = threading.local()


def _session():
    """One HTTP session per thread (sessions are not thread-safe)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def download(url, attempts=4, timeout=180):
    """Return the file's bytes, or ``None`` if the archive does not have it (HTTP 404)."""
    for _ in range(attempts):
        try:
            response = _session().get(url, timeout=timeout)
        except requests.RequestException:
            continue
        if response.status_code == 404:
            return None
        if response.status_code == 200:
            return response.content
    return None


def read_zipped_csv(content, **read_csv_options):
    """Parse the single CSV inside a zip archive held in memory."""
    archive = zipfile.ZipFile(io.BytesIO(content))
    with archive.open(archive.namelist()[0]) as handle:
        return pd.read_csv(handle, **read_csv_options)


def metrics_url(symbol, day):
    """Five-minute open interest, long/short ratios and taker buy/sell ratio for one day."""
    return f"{ARCHIVE}/metrics/{symbol}/{symbol}-metrics-{day}.zip"


def agg_trades_url(symbol, day):
    """Every aggregated trade of one day."""
    return f"{ARCHIVE}/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
