import os
import requests
import pandas as pd
import yfinance as yf

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def main():
    tickers = ["AAPL", "MSFT"]  # luego pondremos tu cartera real

    alerts = []

    for t in tickers:
        df = yf.download(t, period="400d", interval="1d", auto_adjust=True, progress=False)

        if df is None or df.empty:
            continue

        close = float(df["Close"].iloc[-1])
        ma200 = float(df["Close"].rolling(200).mean().iloc[-1])

        if close < ma200:
            alerts.append(f"🚨 {t} por debajo de MA200")

    if alerts:
        msg = "📊 Alertas técnicas\n\n" + "\n".join(alerts)
        send_telegram(msg)
    else:
        print("No alerts today")

if __name__ == "__main__":
    main()
