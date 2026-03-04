import os
import math
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ----------------------------
# TU CARTERA USA (internos)
# ----------------------------
CARTERA_USA = [
    "BAC","PLTR","QBTS","OKLO","RKLB","NBIS","IREN","ZETA",
    "OPEN","EOSE","NVTS","CIFR","NUAI","CAN","ONDS",
    "SKYT","PL","ADUR","RDW","ASST",
    "SATL","IBRX","VG","PRME","ATAI","TMDX","AEHR"
]

# ----------------------------
# SCOUTING EXTERNO
# ----------------------------
SCOUTING_EXT = [
    "SOUN","BBAI","AI","INOD","ALKT","DCBO",
    "LUNR","ASTS","SPIR","KTOS","AVAV",
    "ENVX","STEM","RUN","FLNC",
    "CRNX","IOVA","ARDX","MRSN","SDGR",
    "AMSC","MP","CLSK"
]

UNIVERSO = sorted(set(CARTERA_USA + SCOUTING_EXT))

NEAR_MA200_PCT = 2.0
BREAKOUT_LOOKBACK = 63
ATR_PERIOD = 14
MAX_TG_LEN = 3800
HISTORY_PATH = "data/history.csv"
TZ = ZoneInfo("Europe/Madrid")


# ============================================================
# TELEGRAM (Markdown + safe chunking)
# ============================================================

def _split_markdown_safe(text: str, max_len: int) -> list[str]:
    """
    Split text into chunks <= max_len, but never split while inside a ``` code block.
    """
    lines = (text or "").splitlines(True)  # keep \n
    chunks = []
    buf = ""
    in_code = False

    def is_fence(line: str) -> bool:
        return line.lstrip().startswith("```")

    for line in lines:
        if is_fence(line):
            in_code = not in_code

        if len(buf) + len(line) > max_len:
            if in_code:
                # Avoid splitting mid-code block
                if buf:
                    chunks.append(buf.rstrip("\n"))
                    buf = ""
                buf += line
            else:
                if buf:
                    chunks.append(buf.rstrip("\n"))
                buf = line
        else:
            buf += line

    if buf.strip():
        chunks.append(buf.rstrip("\n"))

    return chunks


def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID en secrets.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chat_id = str(TELEGRAM_CHAT_ID).strip()

    chunks = _split_markdown_safe(text, MAX_TG_LEN)

    for ch in chunks:
        payload = {
            "chat_id": chat_id,
            "text": ch,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


# ============================================================
# UTILS
# ============================================================

def _scalar(x):
    if hasattr(x, "iloc"):
        x = x.iloc[0]
    return x

def _to_float_or_none(x):
    try:
        x = _scalar(x)
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0

def _safe_round(x, nd=2):
    return None if x is None else round(x, nd)

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ============================================================
# INDICATORS
# ============================================================

def compute_atr_pct(df, period=14):
    if df is None or df.empty or "High" not in df or "Low" not in df or "Close" not in df:
        return None
    if len(df) < period + 2:
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = _to_float_or_none(tr.rolling(period).mean().iloc[-1])
    c = _to_float_or_none(close.iloc[-1])
    if atr is None or c is None or c == 0:
        return None
    return (atr / c) * 100.0

def compute_drawdown_pct(df, lookback=63):
    if df is None or df.empty or "Close" not in df:
        return None
    if len(df) < lookback:
        return None
    closes = df["Close"].iloc[-lookback:]
    peak = _to_float_or_none(closes.max())
    last = _to_float_or_none(closes.iloc[-1])
    if peak is None or last is None or peak == 0:
        return None
    return (last / peak - 1.0) * 100.0

def compute_return_pct(df, lookback=20):
    if df is None or df.empty or "Close" not in df:
        return None
    if len(df) < lookback + 1:
        return None
    last = _to_float_or_none(df["Close"].iloc[-1])
    prev = _to_float_or_none(df["Close"].iloc[-(lookback + 1)])
    if last is None or prev in (None, 0):
        return None
    return (last / prev - 1.0) * 100.0


# ============================================================
# CORE ANALYSIS
# ============================================================

def analyze_ticker(ticker: str):
    df = yf.download(
        ticker,
        period="650d",
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

    dist_ma50 = _pct(close, ma50)
    dist_ma200 = _pct(close, ma200)

    breakout_3m = False
    if "High" in df and len(df) >= BREAKOUT_LOOKBACK + 1:
        recent_high = _to_float_or_none(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
        if recent_high is not None:
            breakout_3m = close > recent_high

    atr_pct = compute_atr_pct(df, ATR_PERIOD)
    dd_3m = compute_drawdown_pct(df, BREAKOUT_LOOKBACK)
    ret_1m = compute_return_pct(df, 21)
    ret_3m = compute_return_pct(df, 63)

    ma50_gt_ma200 = (ma50 is not None and ma200 is not None and ma50 > ma200)

    score = 0
    if ma200 and close > ma200: score += 35
    if ma50 and close > ma50: score += 20
    if ma50_gt_ma200: score += 20
    if breakout_3m: score += 25
    score = min(score, 100)

    return {
        "ticker": ticker,
        "close": close,
        "ma50": ma50,
        "ma200": ma200,
        "dist_ma50_pct": dist_ma50,
        "dist_ma200_pct": dist_ma200,
        "ma50_gt_ma200": ma50_gt_ma200,
        "breakout_3m": breakout_3m,
        "atr_pct": atr_pct,
        "dd_3m_pct": dd_3m,
        "ret_1m_pct": ret_1m,
        "ret_3m_pct": ret_3m,
        "score": score,
    }


# ============================================================
# HISTORY
# ============================================================

def ensure_history_dir():
    d = os.path.dirname(HISTORY_PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_last_snapshot():
    if not os.path.exists(HISTORY_PATH):
        return {}
    last_by_ticker = {}
    try:
        with open(HISTORY_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker")
                if t:
                    last_by_ticker[t] = row
    except Exception:
        return {}
    return last_by_ticker

def append_history(date_str: str, rows: list[dict]):
    ensure_history_dir()
    file_exists = os.path.exists(HISTORY_PATH)

    fields = [
        "date","ticker","close","ma50","ma200",
        "dist_ma50_pct","dist_ma200_pct",
        "ma50_gt_ma200","breakout_3m",
        "atr_pct","dd_3m_pct","ret_1m_pct","ret_3m_pct",
        "score"
    ]

    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for r in rows:
            writer.writerow({
                "date": date_str,
                "ticker": r["ticker"],
                "close": _safe_round(r["close"], 4),
                "ma50": _safe_round(r["ma50"], 4),
                "ma200": _safe_round(r["ma200"], 4),
                "dist_ma50_pct": _safe_round(r["dist_ma50_pct"], 3),
                "dist_ma200_pct": _safe_round(r["dist_ma200_pct"], 3),
                "ma50_gt_ma200": int(bool(r["ma50_gt_ma200"])),
                "breakout_3m": int(bool(r["breakout_3m"])),
                "atr_pct": _safe_round(r["atr_pct"], 3),
                "dd_3m_pct": _safe_round(r["dd_3m_pct"], 3),
                "ret_1m_pct": _safe_round(r["ret_1m_pct"], 3),
                "ret_3m_pct": _safe_round(r["ret_3m_pct"], 3),
                "score": int(r["score"]),
            })


# ============================================================
# STRATEGY HELPERS
# ============================================================

def exposure_continuous(pct_over_ma200, pct_ma50_over_ma200, avg_score, n_alerts, n_total):
    base = 0.55 * pct_over_ma200 + 0.35 * pct_ma50_over_ma200 + 0.10 * avg_score
    if n_total > 0:
        base -= 35.0 * (n_alerts / n_total)
    return clamp(base, 0, 100)

def regime_text(pct_over_ma200, pct_ma50_over_ma200, avg_score):
    if pct_over_ma200 >= 80 and pct_ma50_over_ma200 >= 60 and avg_score >= 60:
        return "Riesgo-on (estructura fuerte)"
    if pct_over_ma200 >= 65 and avg_score >= 50:
        return "Constructivo (pero selectivo)"
    if pct_over_ma200 >= 50:
        return "Mixto / lateral (exigente)"
    return "Riesgo-off (estructura dañada)"


# ============================================================
# FORMATTING (mini-tarjeta por ticker)
# ============================================================

def mini_card(r: dict) -> str:
    score = r["score"]
    icon = "🟢" if score >= 70 else ("🟡" if score >= 40 else "🔴")

    d200 = r["dist_ma200_pct"]
    d50 = r["dist_ma50_pct"]
    atr = r["atr_pct"]
    dd = r["dd_3m_pct"]
    r3m = r["ret_3m_pct"]

    d200s = f"{d200:+.1f}%" if d200 is not None else "n/a"
    d50s  = f"{d50:+.1f}%" if d50 is not None else "n/a"
    atrs  = f"{atr:.1f}%" if atr is not None else "n/a"
    dds   = f"{dd:.1f}%" if dd is not None else "n/a"
    r3ms  = f"{r3m:+.1f}%" if r3m is not None else "n/a"

    flags = []
    flags.append("MA50>MA200" if r["ma50_gt_ma200"] else "MA50<MA200")
    if r["breakout_3m"]:
        flags.append("Breakout")

    return (
        f"{icon} *{r['ticker']}* — *{score}/100*\n"
        f"   MA200 {d200s} | MA50 {d50s}\n"
        f"   ATR {atrs} | DD3M {dds} | R3M {r3ms}\n"
        f"   {' | '.join(flags)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    today = datetime.now(TZ).date().isoformat()

    rows = []
    failed = []
    for t in UNIVERSO:
        r = analyze_ticker(t)
        if r:
            rows.append(r)
        else:
            failed.append(t)

    if not rows:
        send_telegram("⚠️ Informe diario: no pude descargar/interpretar datos para ningún ticker.")
        return

    # Separación: cartera vs scouting
    rows_cartera = [r for r in rows if r["ticker"] in set(CARTERA_USA)]
    rows_scout = [r for r in rows if r["ticker"] in set(SCOUTING_EXT)]

    prev = load_last_snapshot()
    append_history(today, rows)

    # Agregados SOLO cartera (para lectura de riesgo/exposición)
    over200 = [r for r in rows_cartera if r["ma200"] and r["close"] > r["ma200"]]
    under200 = [r for r in rows_cartera if r["ma200"] and r["close"] < r["ma200"]]
    ma50_over200 = [r for r in rows_cartera if r["ma50_gt_ma200"]]
    breakouts_cartera = [r for r in rows_cartera if r["breakout_3m"]]
    near200 = [r for r in rows_cartera if r["dist_ma200_pct"] is not None and 0 < r["dist_ma200_pct"] <= NEAR_MA200_PCT]

    # Protección si la cartera estuviera vacía
    denom = max(1, len(rows_cartera))
    pct_over200 = 100.0 * len(over200) / denom
    pct_ma50_over200 = 100.0 * len(ma50_over200) / denom
    avg_score = (sum(r["score"] for r in rows_cartera) / denom) if rows_cartera else 0.0

    # Alertas separadas (cartera)
    alerts_risk = []     # 🚨 ⚠️ 📉
    alerts_momo = []     # 🚀

    # Acciones sugeridas SOLO cartera
    reduce = []
    watch = []
    opp = []

    for r in rows_cartera:
        t = r["ticker"]
        d200 = r["dist_ma200_pct"]
        below200 = (r["ma200"] is not None and r["close"] < r["ma200"])
        ma50lt200 = (r["ma50"] is not None and r["ma200"] is not None and r["ma50"] < r["ma200"])

        if below200:
            alerts_risk.append(f"🚨 {t} bajo MA200")
        if d200 is not None and 0 < d200 <= NEAR_MA200_PCT:
            alerts_risk.append(f"⚠️ {t} cerca MA200 (+{d200:.2f}%)")
        if ma50lt200:
            alerts_risk.append(f"📉 {t} MA50<MA200")
        if r["breakout_3m"]:
            alerts_momo.append(f"🚀 {t} ruptura 3M")

        # acciones
        if below200 and ma50lt200:
            reduce.append(t)
        elif d200 is not None and 0 <= d200 <= 5:
            watch.append(t)
        if r["breakout_3m"] and r["ma50_gt_ma200"] and (r["ma200"] and r["close"] > r["ma200"]):
            opp.append(t)

    # Exposición/régimen (cartera)
    n_alerts_total = len(alerts_risk)  # penalizamos solo “riesgo”
    exposure = exposure_continuous(pct_over200, pct_ma50_over200, avg_score, n_alerts_total, denom)
    regime = regime_text(pct_over200, pct_ma50_over200, avg_score)

    # Movers (sobre todo el universo, por si quieres ver scouting también)
    movers = []
    for r in rows:
        t = r["ticker"]
        prev_row = prev.get(t)
        if prev_row:
            try:
                prev_score = float(prev_row.get("score", "nan"))
                prev_d200 = float(prev_row.get("dist_ma200_pct", "nan"))
            except Exception:
                continue
            ds = r["score"] - prev_score if not math.isnan(prev_score) else 0.0
            dd = (r["dist_ma200_pct"] - prev_d200) if (not math.isnan(prev_d200) and r["dist_ma200_pct"] is not None) else None
            movers.append((t, ds, dd))
    movers_sorted = sorted(movers, key=lambda x: abs(x[1]), reverse=True)[:8]

    # Top/Bottom: SOLO cartera (más coherente para gestionar)
    top = sorted(rows_cartera, key=lambda x: x["score"], reverse=True)[:6]
    bottom = sorted(rows_cartera, key=lambda x: x["score"])[:6]

    # Scouting: oportunidades (por score alto o breakout)
    scout_breakouts = [r for r in rows_scout if r["breakout_3m"]]
    scout_top = sorted(rows_scout, key=lambda x: x["score"], reverse=True)[:6]

    # --------- Build report ----------
    lines = []
    lines.append(f"📊 *INFORME COMPLETO* ({today})")
    lines.append("")
    lines.append("*— Universo —*")
    lines.append(f"- Internos (CARTERA USA): {len(rows_cartera)} | Scouting (SCOUTING EXT): {len(rows_scout)} | Total: {len(rows)} (sin datos: {len(failed)})")
    lines.append("")
    lines.append("*— Agregado (solo cartera) —*")
    lines.append(f"- % sobre MA200: {pct_over200:.0f}% ({len(over200)}/{denom})")
    lines.append(f"- % MA50>MA200: {pct_ma50_over200:.0f}% ({len(ma50_over200)}/{denom})")
    lines.append(f"- Score medio: {avg_score:.0f}/100")
    lines.append(f"- Breakouts 3M (cartera): {len(breakouts_cartera)} | Cerca MA200 (≤{NEAR_MA200_PCT:.1f}%): {len(near200)} | Bajo MA200: {len(under200)}")
    lines.append("")
    lines.append("*— Lectura estratégica (solo cartera) —*")
    lines.append(f"- Régimen: {regime}")
    lines.append(f"- Exposición recomendada (semi-cuanti): *{exposure:.0f}%*")
    if len(under200) >= max(2, int(0.25 * denom)):
        lines.append("- Lectura: daño estructural relevante (prioridad = proteger y recortar riesgo).")
    elif len(near200) >= max(2, int(0.20 * denom)):
        lines.append("- Lectura: varias posiciones en zona crítica (gestionar tamaño / evitar añadir sin confirmación).")
    else:
        lines.append("- Lectura: estructura razonable; se puede ser selectivo con adds en setups fuertes.")
    lines.append("")
    lines.append("*— Acciones sugeridas (solo cartera) —*")
    lines.append(f"- Reducir/recortar (bajo MA200 + MA50<MA200): {', '.join(sorted(set(reduce))) if reduce else 'n/a'}")
    lines.append(f"- Vigilar (0%..+5% sobre MA200): {', '.join(sorted(set(watch))) if watch else 'n/a'}")
    lines.append(f"- Oportunidades (breakout + MA50>MA200 + sobre MA200): {', '.join(sorted(set(opp))) if opp else 'n/a'}")

    if movers_sorted:
        lines.append("")
        lines.append("*— Cambios vs última ejecución (score) —*")
        for t, ds, dd in movers_sorted:
            dd_txt = f", ΔvsMA200 {dd:+.2f}pp" if dd is not None else ""
            sign = "+" if ds >= 0 else ""
            lines.append(f"- {t}: Δscore {sign}{ds:.0f}{dd_txt}")

    lines.append("")
    lines.append("*— Top 6 (cartera) —*")
    if top:
        for r in top:
            lines.append(mini_card(r))
            lines.append("")
    else:
        lines.append("n/a")
        lines.append("")

    lines.append("*— Bottom 6 (cartera) —*")
    if bottom:
        for r in bottom:
            lines.append(mini_card(r))
            lines.append("")
    else:
        lines.append("n/a")
        lines.append("")

    lines.append("*— Scouting (ideas, no acciones) —*")
    if scout_breakouts:
        tickers = ", ".join([r["ticker"] for r in sorted(scout_breakouts, key=lambda x: x["score"], reverse=True)[:10]])
        lines.append(f"- Rupturas 3M: {tickers}")
    else:
        lines.append("- Rupturas 3M: n/a")

    if scout_top:
        lines.append("- Top 6 scouting por score:")
        for r in scout_top:
            lines.append(mini_card(r))
            lines.append("")
    else:
        lines.append("- Top 6 scouting por score: n/a")
        lines.append("")

    if failed:
        lines.append("")
        lines.append("*— Sin datos / error —*")
        lines.append(", ".join(failed))

    send_telegram("\n".join(lines).rstrip())

    # --------- ALERTAS (mensaje separado, con leyenda) ----------
    alert_lines = []
    if alerts_risk:
        alert_lines.append("🚨 *ALERTAS (Riesgo / Control — solo cartera)*")
        alert_lines.append("_Leyenda: 🚨 bajo MA200 | ⚠️ cerca MA200 | 📉 MA50<MA200_")
        alert_lines.append("")
        alert_lines.extend(alerts_risk)

    if alerts_momo:
        if alert_lines:
            alert_lines.append("")
        alert_lines.append("🚀 *ALERTAS (Momentum — solo cartera)*")
        alert_lines.append("_Leyenda: 🚀 ruptura de máximos ~3 meses_")
        alert_lines.append("")
        alert_lines.extend(alerts_momo)

    if alert_lines:
        send_telegram("\n".join(alert_lines))


if __name__ == "__main__":
    main()
