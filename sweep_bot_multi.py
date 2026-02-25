"""
RTH Sweep Strategy - Multi-Ticker Telegram Alert Bot (Alpaca Edition)
======================================================================
Monitorea múltiples acciones al mismo tiempo.

Requirements:
    pip install alpaca-py requests schedule pytz pandas

Setup:
    1. Rellena API_KEY y SECRET_KEY con tus claves de Alpaca Paper Trading
    2. Rellena BOT_TOKEN y CHAT_ID de Telegram
    3. Agrega o quita tickers en la lista TICKERS
    4. Run: python sweep_bot_multi.py
"""

import time
import logging
import requests
import schedule
import pytz
import pandas as pd
from datetime import datetime, time as dtime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================
# CONFIG — edita estos valores
# ============================================================
ALPACA_API_KEY    = "PKUXXRLFZKXYKM4VJOJKHWV2IN"
ALPACA_SECRET_KEY = "7td9mxsp6UNTriDHKfrw5UsbgHVMPey4esNYTg54EsJH"

BOT_TOKEN  = "8656395240:AAHPx8pU0VucN6eWXhnE2h3VyExK0yWXobU"
CHAT_ID    = 6431321918

# ── Agrega o quita los tickers que quieras monitorear ───────
TICKERS = ["QQQ", "SPY", "AAPL", "TSLA", "NVDA","MSFT","META","GOOG","AMZN"]

TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)
TP_PCT       = 0.02
USE_SL       = False
SL_PCT       = 0.01
CONFIRM_BARS = 5

RTH_START = dtime(9, 30)
RTH_END   = dtime(16, 0)
TIMEZONE  = pytz.timezone("America/New_York")
# ============================================================

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Guarda el último timestamp de señal por ticker para no repetir alertas
_last_signal_time: dict = {ticker: None for ticker in TICKERS}


def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        log.info(f"Telegram enviado: {message[:60]}")
    except Exception as e:
        log.error(f"Telegram error: {e}")


def is_rth(dt: datetime) -> bool:
    t = dt.astimezone(TIMEZONE).time()
    return RTH_START <= t < RTH_END


def fetch_data(ticker: str) -> pd.DataFrame:
    """Descarga los últimos 7 días de velas para un ticker."""
    end   = datetime.now(TIMEZONE)
    start = end - timedelta(days=7)

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        feed="iex"
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if df.empty:
        raise ValueError(f"Alpaca no devolvió datos para {ticker}")

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df.index = pd.DatetimeIndex(df.index).tz_convert(TIMEZONE)
    df.rename(columns={"open": "Open", "high": "High",
                        "low": "Low", "close": "Close",
                        "volume": "Volume"}, inplace=True)
    df["in_rth"] = df.index.map(is_rth)
    df["date"]   = df.index.date
    return df


def compute_prev_rth_levels(df: pd.DataFrame) -> pd.DataFrame:
    rth_df = df[df["in_rth"]].copy()
    daily = rth_df.groupby("date").agg(
        rth_high=("High", "max"),
        rth_low=("Low",  "min")
    )
    daily["prev_rth_high"] = daily["rth_high"].shift(1)
    daily["prev_rth_low"]  = daily["rth_low"].shift(1)
    df = df.join(daily[["prev_rth_high", "prev_rth_low"]], on="date")
    return df


def find_signals(df: pd.DataFrame) -> list:
    signals     = []
    hi_raid_bar = None
    lo_raid_bar = None
    prev_date   = None

    for i, (ts, row) in enumerate(df.iterrows()):
        if row["date"] != prev_date:
            hi_raid_bar = None
            lo_raid_bar = None
            prev_date   = row["date"]

        if not row["in_rth"]:
            continue

        ph = row["prev_rth_high"]
        pl = row["prev_rth_low"]
        if pd.isna(ph) or pd.isna(pl):
            continue

        # Step 1: Raid
        if row["High"] > ph and hi_raid_bar is None:
            hi_raid_bar = i
        if row["Low"] < pl and lo_raid_bar is None:
            lo_raid_bar = i

        # Step 2: Reclaim → SHORT
        if (hi_raid_bar is not None and
                (i - hi_raid_bar) <= CONFIRM_BARS and
                row["Close"] < ph):
            signals.append({
                "time":      ts,
                "type":      "SHORT",
                "price":     row["Close"],
                "prev_high": ph,
                "prev_low":  pl,
                "tp":        row["Close"] * (1 - TP_PCT),
                "sl":        row["Close"] * (1 + SL_PCT) if USE_SL else None,
            })
            hi_raid_bar = None

        # Step 2: Reclaim → LONG
        if (lo_raid_bar is not None and
                (i - lo_raid_bar) <= CONFIRM_BARS and
                row["Close"] > pl):
            signals.append({
                "time":      ts,
                "type":      "LONG",
                "price":     row["Close"],
                "prev_high": ph,
                "prev_low":  pl,
                "tp":        row["Close"] * (1 + TP_PCT),
                "sl":        row["Close"] * (1 - SL_PCT) if USE_SL else None,
            })
            lo_raid_bar = None

    return signals


def check_ticker(ticker: str):
    """Revisa un ticker y manda alerta si hay señal nueva."""
    global _last_signal_time
    log.info(f"Revisando {ticker}...")

    try:
        df      = fetch_data(ticker)
        df      = compute_prev_rth_levels(df)
        signals = find_signals(df)
    except Exception as e:
        log.error(f"Error en {ticker}: {e}")
        return

    if not signals:
        return

    latest = signals[-1]
    if latest["time"] == _last_signal_time[ticker]:
        return

    _last_signal_time[ticker] = latest["time"]
    s = latest
    direction = "🟢 LONG" if s["type"] == "LONG" else "🔴 SHORT"
    sl_line   = f"🛑 Stop Loss: `${s['sl']:.2f}`" if s["sl"] else "🛑 Stop Loss: OFF"

    msg = (
        f"*{ticker} — Sweep Signal*\n"
        f"──────────────────\n"
        f"{direction}\n"
        f"⏰ `{s['time'].strftime('%Y-%m-%d %H:%M %Z')}`\n"
        f"💵 Entry: `${s['price']:.2f}`\n"
        f"🎯 Take Profit: `${s['tp']:.2f}` ({TP_PCT*100:.1f}%)\n"
        f"{sl_line}\n"
        f"──────────────────\n"
        f"📊 Prev RTH High: `${s['prev_high']:.2f}`\n"
        f"📊 Prev RTH Low:  `${s['prev_low']:.2f}`"
    )
    send_telegram(msg)


def check_all_tickers():
    """Revisa todos los tickers de la lista."""
    for ticker in TICKERS:
        check_ticker(ticker)
        time.sleep(1)  # pequeña pausa para no saturar la API


def main():
    tickers_str = ", ".join(f"`{t}`" for t in TICKERS)
    log.info(f"Bot iniciado | Monitoreando: {', '.join(TICKERS)}")
    send_telegram(
        f"🤖 *Sweep Bot iniciado* (Alpaca)\n"
        f"Monitoreando: {tickers_str}"
    )

    check_all_tickers()
    schedule.every(1).minutes.do(check_all_tickers)

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()
