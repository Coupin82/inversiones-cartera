import os
import math
import requests
import yfinance as yf

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ====== CONFIG ======
TICKERS = [
    "BAC","PLTR","QBTS","OKLO","RKLB","NBIS","IREN","ZETA",
    "OPEN","EOSE","NVTS","CIFR","NUAI","CAN","ONDS",
    "SKYT","PL","ADUR","RDW","ASST",
    "SATL","IBRX","VG","PRME","ATAI","TMDX"
]
NEAR_MA200_PCT = 2.0
BREAKOUT_LOOKBACK = 63
# ====================


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


def scalar(x):
    if hasattr(x, "iloc"):
        x = x.iloc[0]
    return float(x)


def analyze(ticker):
    df = yf.download(ticker, period="500d", interval="1d",
                     auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None

    close = scalar(df["Close"].iloc[-1])
    ma50 = scalar(df["Close"].rolling(50).mean().iloc[-1])
    ma200 = scalar(df["Close"].rolling(200).mean().iloc[-1])

    dist200 = (close / ma200 - 1) * 100 if ma200 else None
    breakout = False

    if len(df) > BREAKOUT_LOOKBACK:
        recent_high = scalar(
            df["High"].iloc[-(BREAKOUT_LOOKBACK+1):-1].max()
        )
        breakout = close > recent_high

    # Score
    score = 0
    if close > ma200: score += 35
    if close > ma50: score += 20
    if ma50 > ma200: score += 20
    if breakout: score += 25

    return {
        "ticker": ticker,
        "close": close,
        "ma50": ma50,
        "ma200": ma200,
        "dist200": dist200,
        "breakout": breakout,
        "score": score
    }


def main():
    data = []
    alerts = []

    for t in TICKERS:
        r = analyze(t)
        if r:
            data.append(r)

            if r["close"] < r["ma200"]:
                alerts.append(f"🚨 {t} bajo MA200")

            if 0 < r["dist200"] <= NEAR_MA200_PCT:
                alerts.append(f"⚠️ {t} cerca MA200 (+{r['dist200']:.1f}%)")

            if r["ma50"] < r["ma200"]:
                alerts.append(f"📉 {t} MA50 < MA200")

            if r["breakout"]:
                alerts.append(f"🚀 {t} ruptura 3M")

    if not data:
        send_telegram("⚠️ No hay datos.")
        return

    over200 = [d for d in data if d["close"] > d["ma200"]]
    ma50over200 = [d for d in data if d["ma50"] > d["ma200"]]

    pct_over200 = 100 * len(over200) / len(data)
    pct_ma50over200 = 100 * len(ma50over200) / len(data)

    avg_score = sum(d["score"] for d in data) / len(data)

    # Exposición sugerida
    if pct_over200 > 80:
        exposure = "Alta (80-100%)"
    elif pct_over200 > 60:
        exposure = "Media (60-80%)"
    else:
        exposure = "Defensiva (<60%)"

    top = sorted(data, key=lambda x: x["score"], reverse=True)[:5]
    bottom = sorted(data, key=lambda x: x["score"])[:5]

    def fmt(d):
        return f"{d['ticker']} ({d['score']})"

    summary = (
        "📊 INFORME TÉCNICO COMPLETO\n\n"
        f"Valores: {len(data)}\n"
        f"% sobre MA200: {pct_over200:.0f}%\n"
        f"% MA50>MA200: {pct_ma50over200:.0f}%\n"
        f"Score medio: {avg_score:.0f}\n"
        f"Exposición sugerida: {exposure}\n\n"
        "Top técnicos:\n"
        + ", ".join(fmt(d) for d in top)
        + "\n\nBottom técnicos:\n"
        + ", ".join(fmt(d) for d in bottom)
    )

    send_telegram(summary)

    if alerts:
        send_telegram("🚨 ALERTAS\n\n" + "\n".join(alerts))


if __name__ == "__main__":
    main()
