import os
import requests
import yfinance as yf

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ====== CONFIG ======
TICKERS = [
    "AAPL", "MSFT",  # <-- sustituye por tu cartera real
]

NEAR_MA200_PCT = 2.0      # “cerca de perder MA200” si está entre 0% y +2%
BREAKOUT_LOOKBACK = 63    # ~3 meses bursátiles
ALERTS_ONLY_IF_TRIGGERED = True  # si True, solo manda el mensaje de alertas si hay alertas
# ====================


def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID en secrets.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chat_id = str(TELEGRAM_CHAT_ID).strip()

    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }

    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


def _to_float(x):
    # yfinance a veces devuelve Series (MultiIndex). Esto lo normaliza.
    if hasattr(x, "iloc"):
        x = x.iloc[0]
    return float(x)


def analyze_ticker(ticker: str):
    df = yf.download(
        ticker,
        period="500d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if df is None or df.empty or "Close" not in df:
        return None

    close_last = df["Close"].iloc[-1]
    close = _to_float(close_last)

    ma50_s = df["Close"].rolling(50).mean()
    ma200_s = df["Close"].rolling(200).mean()

    ma50_last = ma50_s.iloc[-1]
    ma200_last = ma200_s.iloc[-1]

    ma50 = _to_float(ma50_last) if ma50_last == ma50_last else None   # NaN check
    ma200 = _to_float(ma200_last) if ma200_last == ma200_last else None

    # Distancias
    dist_ma200_pct = None
    dist_ma50_pct = None
    if ma200 and ma200 != 0:
        dist_ma200_pct = (close / ma200 - 1.0) * 100.0
    if ma50 and ma50 != 0:
        dist_ma50_pct = (close / ma50 - 1.0) * 100.0

    # Breakout 3M (close > max high lookback)
    breakout_3m = False
    if len(df) >= BREAKOUT_LOOKBACK + 1 and "High" in df:
        recent_high = df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max()
        recent_high = _to_float(recent_high) if hasattr(recent_high, "iloc") else float(recent_high)
        breakout_3m = close > recent_high

    # Scoring simple (puedes endurecerlo luego)
    score = 0
    if ma200 and close > ma200: score += 35
    if ma50 and close > ma50: score += 20
    if ma50 and ma200 and ma50 > ma200: score += 20
    if breakout_3m: score += 25
    score = min(score, 100)

    return {
        "ticker": ticker,
        "close": close,
        "ma50": ma50,
        "ma200": ma200,
        "dist_ma50_pct": dist_ma50_pct,
        "dist_ma200_pct": dist_ma200_pct,
        "breakout_3m": breakout_3m,
        "score": score,
    }


def main():
    rows = []
    for t in TICKERS:
        r = analyze_ticker(t)
        if r:
            rows.append(r)

    if not rows:
        send_telegram("⚠️ Daily Alerts: no he podido descargar datos (lista vacía o error de descarga).")
        return

    # ====== RESUMEN DIARIO (SIEMPRE) ======
    over_ma200 = [r for r in rows if r["ma200"] and r["close"] > r["ma200"]]
    under_ma200 = [r for r in rows if r["ma200"] and r["close"] < r["ma200"]]
    near_ma200 = [r for r in rows if r["dist_ma200_pct"] is not None and 0 < r["dist_ma200_pct"] <= NEAR_MA200_PCT]
    breakouts = [r for r in rows if r["breakout_3m"]]

    avg_score = sum(r["score"] for r in rows) / len(rows)
    pct_over_ma200 = 100.0 * len(over_ma200) / len(rows)

    top = sorted(rows, key=lambda x: x["score"], reverse=True)[:5]
    bottom = sorted(rows, key=lambda x: x["score"])[:5]

    def fmt_line(r):
        d200 = r["dist_ma200_pct"]
        d200s = f"{d200:+.1f}%" if d200 is not None else "n/a"
        b = "🚀" if r["breakout_3m"] else ""
        return f"- {r['ticker']}: score {r['score']}/100 | vs MA200 {d200s} {b}"

    summary_msg = (
        "📊 <b>Resumen diario cartera</b>\n"
        f"- Nº valores: {len(rows)}\n"
        f"- % sobre MA200: {pct_over_ma200:.0f}% ({len(over_ma200)}/{len(rows)})\n"
        f"- Score medio: {avg_score:.0f}/100\n"
        f"- Breakouts 3M: {len(breakouts)}\n"
        "\n<b>Top</b>\n" + "\n".join(fmt_line(r) for r in top) +
        "\n\n<b>Bottom</b>\n" + "\n".join(fmt_line(r) for r in bottom)
    )
    # Nota: parse_mode HTML no lo estamos usando aquí para evitar problemas; si quieres negritas, lo activamos.
    # De momento lo enviamos como texto plano:
    send_telegram(summary_msg.replace("<b>", "").replace("</b>", ""))

    # ====== ALERTAS (SOLO SI HAY) ======
    alerts = []

    for r in rows:
        t = r["ticker"]
        if r["ma200"] and r["close"] < r["ma200"]:
            alerts.append(f"🚨 {t} por debajo de MA200")
        if r["dist_ma200_pct"] is not None and 0 < r["dist_ma200_pct"] <= NEAR_MA200_PCT:
            alerts.append(f"⚠️ {t} cerca de MA200 ({r['dist_ma200_pct']:.2f}%)")
        if r["ma50"] and r["ma200"] and r["ma50"] < r["ma200"]:
            alerts.append(f"📉 {t} MA50 < MA200 (tendencia débil)")
        if r["breakout_3m"]:
            alerts.append(f"🚀 {t} breakout 3M")

    if alerts or not ALERTS_ONLY_IF_TRIGGERED:
        alerts_msg = "🚨 Alertas\n\n" + ("\n".join(alerts) if alerts else "Sin alertas hoy.")
        send_telegram(alerts_msg)


if __name__ == "__main__":
    main()
