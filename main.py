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
NEAR_MA200_PCT = 2.0          # “cerca de perder MA200” si 0%..+2%
BREAKOUT_LOOKBACK = 63        # ~3 meses bursátiles
MAX_TG_LEN = 3800             # margen bajo 4096
# ====================


def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID en secrets.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chat_id = str(TELEGRAM_CHAT_ID).strip()

    # Telegram tiene límite de longitud por mensaje: troceamos si hace falta
    chunks = []
    s = text
    while len(s) > MAX_TG_LEN:
        cut = s.rfind("\n", 0, MAX_TG_LEN)
        if cut == -1:
            cut = MAX_TG_LEN
        chunks.append(s[:cut])
        s = s[cut:].lstrip("\n")
    chunks.append(s)

    for ch in chunks:
        payload = {"chat_id": chat_id, "text": ch, "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


def _scalar(x):
    """Convierte Series/array (MultiIndex) a escalar."""
    if hasattr(x, "iloc"):
        x = x.iloc[0]
    return x


def _to_float_or_none(x):
    """Convierte a float; devuelve None si NaN/inf o no convertible."""
    try:
        x = _scalar(x)
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def analyze_ticker(ticker: str):
    df = yf.download(
        ticker,
        period="550d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty or "Close" not in df:
        return None

    close = _to_float_or_none(df["Close"].iloc[-1])
    if close is None:
        return None

    ma50 = _to_float_or_none(df["Close"].rolling(50).mean().iloc[-1])
    ma200 = _to_float_or_none(df["Close"].rolling(200).mean().iloc[-1])

    dist200 = None
    dist50 = None
    if ma200:
        dist200 = (close / ma200 - 1.0) * 100.0
    if ma50:
        dist50 = (close / ma50 - 1.0) * 100.0

    breakout_3m = False
    if "High" in df and len(df) >= BREAKOUT_LOOKBACK + 1:
        recent_high = df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max()
        recent_high = _to_float_or_none(recent_high)
        if recent_high:
            breakout_3m = close > recent_high

    # Score 0-100 (simple pero útil; lo endurecemos si quieres)
    score = 0
    if ma200 and close > ma200:
        score += 35
    if ma50 and close > ma50:
        score += 20
    if ma50 and ma200 and ma50 > ma200:
        score += 20
    if breakout_3m:
        score += 25
    score = min(score, 100)

    return {
        "ticker": ticker,
        "close": close,
        "ma50": ma50,
        "ma200": ma200,
        "dist_ma50_pct": dist50,
        "dist_ma200_pct": dist200,
        "breakout_3m": breakout_3m,
        "score": score,
    }


def exposure_suggestion(pct_over_ma200: float, pct_ma50_over_ma200: float, avg_score: float) -> str:
    # Regla táctica sencilla: mezcla estructura + tendencia + momentum (score)
    if pct_over_ma200 >= 80 and pct_ma50_over_ma200 >= 65 and avg_score >= 65:
        return "ALTA (80–100%)"
    if pct_over_ma200 >= 65 and pct_ma50_over_ma200 >= 50 and avg_score >= 55:
        return "MEDIA (60–80%)"
    if pct_over_ma200 >= 50 and avg_score >= 45:
        return "BAJA (40–60%)"
    return "DEFENSIVA (0–40%)"


def regime_label(pct_over_ma200: float, pct_ma50_over_ma200: float, avg_score: float) -> str:
    # Lectura “gestor”
    if pct_over_ma200 >= 80 and pct_ma50_over_ma200 >= 60 and avg_score >= 60:
        return "Riesgo-on (estructura fuerte)"
    if pct_over_ma200 >= 65 and avg_score >= 50:
        return "Constructivo (pero selectivo)"
    if pct_over_ma200 >= 50:
        return "Mixto / lateral (más exigente)"
    return "Riesgo-off (estructura dañada)"


def main():
    rows = []
    errors = []

    for t in TICKERS:
        r = analyze_ticker(t)
        if r:
            rows.append(r)
        else:
            errors.append(t)

    if not rows:
        send_telegram("⚠️ Informe diario: no pude descargar/interpretar datos para ningún ticker.")
        return

    # Agregados
    over_ma200 = [r for r in rows if r["ma200"] and r["close"] > r["ma200"]]
    under_ma200 = [r for r in rows if r["ma200"] and r["close"] < r["ma200"]]
    ma50_over_ma200 = [r for r in rows if r["ma50"] and r["ma200"] and r["ma50"] > r["ma200"]]
    breakouts = [r for r in rows if r["breakout_3m"]]
    near_ma200 = [r for r in rows if r["dist_ma200_pct"] is not None and 0 < r["dist_ma200_pct"] <= NEAR_MA200_PCT]

    pct_over_ma200 = 100.0 * len(over_ma200) / len(rows)
    pct_ma50_over_ma200 = 100.0 * len(ma50_over_ma200) / len(rows)
    avg_score = sum(r["score"] for r in rows) / len(rows)

    exposure = exposure_suggestion(pct_over_ma200, pct_ma50_over_ma200, avg_score)
    regime = regime_label(pct_over_ma200, pct_ma50_over_ma200, avg_score)

    # Listas
    top_score = sorted(rows, key=lambda x: x["score"], reverse=True)[:7]
    bottom_score = sorted(rows, key=lambda x: x["score"])[:7]

    # más cerca de MA200 (peligro por proximidad positiva)
    near_sorted = sorted(
        [r for r in rows if r["dist_ma200_pct"] is not None and r["dist_ma200_pct"] >= 0],
        key=lambda x: x["dist_ma200_pct"]
    )[:8]

    # más fuertes por distancia a MA200
    strong_sorted = sorted(
        [r for r in rows if r["dist_ma200_pct"] is not None],
        key=lambda x: x["dist_ma200_pct"],
        reverse=True
    )[:8]

    def fmt_score(r):
        d200 = r["dist_ma200_pct"]
        d200s = f"{d200:+.1f}%" if d200 is not None else "n/a"
        br = " 🚀" if r["breakout_3m"] else ""
        return f"- {r['ticker']}: score {r['score']}/100 | vs MA200 {d200s}{br}"

    def fmt_short(r):
        d200 = r["dist_ma200_pct"]
        d200s = f"{d200:+.1f}%" if d200 is not None else "n/a"
        return f"- {r['ticker']}: {d200s}"

    # ====== MENSAJE 1: INFORME COMPLETO (técnico + lectura) ======
    strategy_lines = []
    strategy_lines.append(f"Régimen: {regime}")
    strategy_lines.append(f"Exposición sugerida: {exposure}")

    # Interpretación simple
    if len(under_ma200) >= max(3, int(0.25 * len(rows))):
        strategy_lines.append("Lectura: hay daño estructural en parte relevante de la cartera (prioridad = proteger).")
    elif len(near_ma200) >= max(3, int(0.20 * len(rows))):
        strategy_lines.append("Lectura: muchas posiciones en zona de decisión (gestionar tamaño / stops / evitar añadir).")
    else:
        strategy_lines.append("Lectura: estructura razonable; se puede ser selectivo con adds si hay setups claros.")

    if breakouts:
        strategy_lines.append("Nota: hay rupturas 3M → potencial de continuación (mejor en nombres con MA50>MA200).")

    msg_report = (
        "📊 INFORME TÉCNICO (Cartera)\n\n"
        "— Agregado —\n"
        f"- Nº valores analizados: {len(rows)}\n"
        f"- % sobre MA200: {pct_over_ma200:.0f}% ({len(over_ma200)}/{len(rows)})\n"
        f"- % MA50 > MA200: {pct_ma50_over_ma200:.0f}% ({len(ma50_over_ma200)}/{len(rows)})\n"
        f"- Score medio: {avg_score:.0f}/100\n"
        f"- Bajo MA200: {len(under_ma200)}\n"
        f"- Cerca MA200 (≤{NEAR_MA200_PCT:.1f}%): {len(near_ma200)}\n"
        f"- Breakouts 3M: {len(breakouts)}\n\n"
        "— Lectura estratégica —\n"
        + "\n".join(f"- {s}" for s in strategy_lines)
        + "\n\n— Top (score) —\n"
        + ("\n".join(fmt_score(r) for r in top_score) if top_score else "- n/a")
        + "\n\n— Bottom (score) —\n"
        + ("\n".join(fmt_score(r) for r in bottom_score) if bottom_score else "- n/a")
        + "\n\n— Más cerca de MA200 (riesgo) —\n"
        + ("\n".join(fmt_short(r) for r in near_sorted) if near_sorted else "- n/a")
        + "\n\n— Más fuertes vs MA200 —\n"
        + ("\n".join(fmt_short(r) for r in strong_sorted) if strong_sorted else "- n/a")
    )

    # si hay tickers sin datos, lo anotamos al final
    if errors:
        msg_report += "\n\n— Sin datos / error —\n" + ", ".join(errors)

    send_telegram(msg_report)

    # ====== MENSAJE 2: ALERTAS (solo si hay) ======
    alerts = []

    for r in rows:
        t = r["ticker"]
        if r["ma200"] and r["close"] < r["ma200"]:
            alerts.append(f"🚨 {t} por debajo de MA200")
        if r["dist_ma200_pct"] is not None and 0 < r["dist_ma200_pct"] <= NEAR_MA200_PCT:
            alerts.append(f"⚠️ {t} cerca de MA200 (+{r['dist_ma200_pct']:.2f}%)")
        if r["ma50"] and r["ma200"] and r["ma50"] < r["ma200"]:
            alerts.append(f"📉 {t} MA50 < MA200")
        if r["breakout_3m"]:
            alerts.append(f"🚀 {t} ruptura 3M")

    if alerts:
        send_telegram("🚨 ALERTAS\n\n" + "\n".join(alerts))


if __name__ == "__main__":
    main()
