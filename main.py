import os
import requests
import pandas as pd
import yfinance as yf

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


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


def get_last_value(series):
    """Devuelve el último valor aunque venga como Series (MultiIndex)."""
    value = series.iloc[-1]
    if hasattr(value, "iloc"):
        value = value.iloc[0]
    return float(value)


def main():
    tickers = ["AAPL", "MSFT"]  # luego pondremos tu cartera real

    alerts = []

    for t in tickers:
        df = yf.download(
            t,
            period="400d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            continue

        close = get_last_value(df["Close"])
        ma200_series = df["Close"].rolling(200).mean()
        ma200 = get_last_value(ma200_series)

        if close < ma200:
            alerts.append(f"🚨 {t} por debajo de MA200")

    if alerts:
        msg = "📊 Alertas técnicas\n\n" + "\n".join(alerts)
        send_telegram(msg)
    else:
        send_telegram("✅ Sistema ejecutado correctamente. Sin alertas hoy.")


if __name__ == "__main__":
    main()
