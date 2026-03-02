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
# ====================


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text
    }
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


def to_float(x):
    if hasattr(x, "iloc"):
        x = x.iloc[0]
    return float(x)


def analyze_ticker(ticker):
    df = yf.download(ticker, period="400d", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None

    close = to_float(df["Close"].iloc[-1])
    ma200 = to_float(df["Close"].rolling(200).mean().iloc[-1])

    return {
        "ticker": ticker,
        "close": close,
        "ma200": ma200
    }


def main():
    results = []

    for t in TICKERS:
        r = analyze_ticker(t)
        if r:
            results.append(r)

    if not results:
        send_telegram("⚠️ No se pudieron obtener datos.")
        return

    over = [r for r in results if r["close"] > r["ma200"]]

    msg = (
        "📊 Resumen diario\n"
        f"Valores: {len(results)}\n"
        f"Sobre MA200: {len(over)}/{len(results)}"
    )

    send_telegram(msg)


if __name__ == "__main__":
    main()
