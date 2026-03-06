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

CARTERA_USA = [
    "BAC","PLTR","QBTS","OKLO","RKLB","NBIS","IREN","ZETA",
    "OPEN","EOSE","NVTS","CIFR","NUAI","CAN","ONDS",
    "SKYT","PL","ADUR","RDW","ASST",
    "SATL","IBRX","VG","PRME","TMDX","AEHR","SOFI",
    "IOVA",
]

SCOUTING_EXT = [
    "SOUN","BBAI","AI","INOD","ALKT","DCBO",
    "LUNR","ASTS","SPIR","KTOS","AVAV",
    "ENVX","STEM","RUN","FLNC",
    "CRNX","ARDX","MRSN","SDGR",
    "AMSC","MP","CLSK"
]

UNIVERSO = sorted(set(CARTERA_USA + SCOUTING_EXT))

NEAR_MA200_PCT = 2.0                # 0..+2% (cerca de perder MA200)
RECOVER_MA200_PCT = 2.0             # -2%..0% (cerca de recuperar MA200)
BREAKOUT_LOOKBACK = 63              # ~3 meses bursátiles
ATR_PERIOD = 14
MAX_TG_LEN = 3800
HISTORY_PATH = "data/history.csv"
TZ = ZoneInfo("Europe/Madrid")

# A) filtro anti-"muy extendida" para oportunidades (sobre MA200)
OPP_MAX_EXT_MA200_PCT = 12.0        # 0..+12% sobre MA200

# C) buckets de tamaño por ATR%
ATR_BIG_MAX = 3.0
ATR_MED_MAX = 6.0


# ============================================================
# TELEGRAM (Markdown + safe chunking)
# ============================================================

def _split_markdown_safe(text: str, max_len: int) -> list[str]:
    lines = (text or "").splitlines(True)
    chunks, buf = [], ""
    in_code = False

    def is_fence(line: str) -> bool:
        return line.lstrip().startswith("```")

    for line in lines:
        if is_fence(line):
            in_code = not in_code

        if len(buf) + len(line) > max_len:
            if in_code:
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

    for ch in _split_markdown_safe(text, MAX_TG_LEN):
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

def _fmt_pct(x, nd=1, signed=True):
    if x is None:
        return "n/a"
    if signed:
        return f"{x:+.{nd}f}%"
    return f"{x:.{nd}f}%"


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
    prev_close = _to_float_or_none(df["Close"].iloc[-2]) if len(df) >= 2 else None
    if close is None:
        return None

    ma50 = _to_float_or_none(df["Close"].rolling(50).mean().iloc[-1])
    ma200 = _to_float_or_none(df["Close"].rolling(200).mean().iloc[-1])

    dist_ma50 = _pct(close, ma50)
    dist_ma200 = _pct(close, ma200)

    # "Pérdida reciente de MA200" (cruce a la baja en la última sesión)
    recently_lost_ma200 = False
    if ma200 is not None and prev_close is not None:
        recently_lost_ma200 = (prev_close > ma200 and close < ma200)

    # Breakout + calidad (D)
    breakout_3m = False
    breakout_quality_pct = None  # % por encima del max previo 3M
    if "High" in df and len(df) >= BREAKOUT_LOOKBACK + 1:
        recent_high = _to_float_or_none(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
        if recent_high is not None and recent_high > 0:
            breakout_3m = close > recent_high
            breakout_quality_pct = (close / recent_high - 1.0) * 100.0

    atr_pct = compute_atr_pct(df, ATR_PERIOD)
    dd_3m = compute_drawdown_pct(df, BREAKOUT_LOOKBACK)
    ret_1m = compute_return_pct(df, 21)
    ret_3m = compute_return_pct(df, 63)

    # E) retorno 1D para movers
    ret_1d = None
    if prev_close not in (None, 0):
        ret_1d = (close / prev_close - 1.0) * 100.0

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
        "breakout_quality_pct": breakout_quality_pct,
        "atr_pct": atr_pct,
        "dd_3m_pct": dd_3m,
        "ret_1m_pct": ret_1m,
        "ret_3m_pct": ret_3m,
        "ret_1d_pct": ret_1d,
        "score": score,
        "recently_lost_ma200": recently_lost_ma200,
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
        "ma50_gt_ma200","breakout_3m","breakout_quality_pct",
        "atr_pct","dd_3m_pct","ret_1m_pct","ret_3m_pct","ret_1d_pct",
        "score","recently_lost_ma200"
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
                "breakout_quality_pct": _safe_round(r.get("breakout_quality_pct"), 3),
                "atr_pct": _safe_round(r["atr_pct"], 3),
                "dd_3m_pct": _safe_round(r["dd_3m_pct"], 3),
                "ret_1m_pct": _safe_round(r["ret_1m_pct"], 3),
                "ret_3m_pct": _safe_round(r["ret_3m_pct"], 3),
                "ret_1d_pct": _safe_round(r.get("ret_1d_pct"), 3),
                "score": int(r["score"]),
                "recently_lost_ma200": int(bool(r.get("recently_lost_ma200", False))),
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

def exposure_light(exposure_pct: float) -> str:
    if exposure_pct >= 70:
        return "🟢"
    if exposure_pct >= 40:
        return "🟡"
    return "🔴"

def size_bucket_from_atr(atr_pct: float | None) -> str:
    # C) buckets de tamaño por ATR
    if atr_pct is None:
        return "n/a"
    if atr_pct <= ATR_BIG_MAX:
        return "Grande"
    if atr_pct <= ATR_MED_MAX:
        return "Medio"
    return "Pequeño"


# ============================================================
# FORMATTING (mini-tarjeta por ticker)
# ============================================================

def mini_card(r: dict, include_size_hint: bool = False) -> str:
    score = r["score"]
    icon = "🟢" if score >= 70 else ("🟡" if score >= 40 else "🔴")

    d200s = _fmt_pct(r["dist_ma200_pct"], 1, signed=True)
    d50s  = _fmt_pct(r["dist_ma50_pct"], 1, signed=True)
    atrs  = _fmt_pct(r["atr_pct"], 1, signed=False)
    dds   = _fmt_pct(r["dd_3m_pct"], 1, signed=False)
    r3ms  = _fmt_pct(r["ret_3m_pct"], 1, signed=True)

    flags = []
    flags.append("MA50>MA200" if r["ma50_gt_ma200"] else "MA50<MA200")
    if r["breakout_3m"]:
        # D) calidad del breakout
        q = r.get("breakout_quality_pct")
        qtxt = f"{q:+.1f}%" if q is not None else "n/a"
        flags.append(f"Breakout {qtxt}")

    extra = ""
    if include_size_hint:
        extra = f"\n   Tamaño sugerido (ATR): {size_bucket_from_atr(r.get('atr_pct'))}"

    return (
        f"{icon} *{r['ticker']}* — *{score}/100*\n"
        f"   MA200 {d200s} | MA50 {d50s}\n"
        f"   ATR {atrs} | DD3M {dds} | R3M {r3ms}\n"
        f"   {' | '.join(flags)}"
        f"{extra}"
    )

def arrow_delta(x: float) -> str:
    if x is None:
        return ""
    if x > 0:
        return "↑"
    if x < 0:
        return "↓"
    return "→"


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
    set_cartera = set(CARTERA_USA)
    set_scout = set(SCOUTING_EXT)
    rows_cartera = [r for r in rows if r["ticker"] in set_cartera]
    rows_scout = [r for r in rows if r["ticker"] in set_scout]

    prev = load_last_snapshot()
    append_history(today, rows)

    # ========== Agregados SOLO cartera ==========
    denom = max(1, len(rows_cartera))

    over200 = [r for r in rows_cartera if r["ma200"] and r["close"] > r["ma200"]]
    under200 = [r for r in rows_cartera if r["ma200"] and r["close"] < r["ma200"]]
    ma50_over200 = [r for r in rows_cartera if r["ma50_gt_ma200"]]
    breakouts_cartera = [r for r in rows_cartera if r["breakout_3m"]]

    near200 = [
        r for r in rows_cartera
        if r["dist_ma200_pct"] is not None and 0 < r["dist_ma200_pct"] <= NEAR_MA200_PCT
    ]

    recover200 = [
        r for r in rows_cartera
        if r["dist_ma200_pct"] is not None and (-RECOVER_MA200_PCT) <= r["dist_ma200_pct"] <= 0
    ]

    recently_lost = [r for r in rows_cartera if r.get("recently_lost_ma200")]

    pct_over200 = 100.0 * len(over200) / denom
    pct_ma50_over200 = 100.0 * len(ma50_over200) / denom
    avg_score = (sum(r["score"] for r in rows_cartera) / denom) if rows_cartera else 0.0

    # B) breadth por calidad: verdes y rojos
    n_green = sum(1 for r in rows_cartera if r["score"] >= 70)
    pct_green = 100.0 * n_green / denom
    n_red = sum(1 for r in rows_cartera if r["score"] < 40)
    pct_red = 100.0 * n_red / denom

    # Top riesgo por volatilidad (ATR%)
    atr_sorted = sorted(
        [r for r in rows_cartera if r["atr_pct"] is not None],
        key=lambda x: x["atr_pct"],
        reverse=True
    )
    top_atr = atr_sorted[:3]

    # E) movers por retorno 1D (cartera)
    movers_1d = [r for r in rows_cartera if r.get("ret_1d_pct") is not None]
    top_1d = sorted(movers_1d, key=lambda x: x["ret_1d_pct"], reverse=True)[:3]
    bot_1d = sorted(movers_1d, key=lambda x: x["ret_1d_pct"])[:3]

    # ========== Alertas y acciones (solo cartera) ==========
    alerts_risk = []
    alerts_momo = []

    reduce = []
    watch = []

    # A) Oportunidades: más estrictas (anti-extendidas)
    opp_strict = []   # lista de dicts (para mostrar con tamaño sugerido)
    opp_reject_ext = []  # tickers que cumplían todo menos extensión

    for r in rows_cartera:
        t = r["ticker"]
        d200 = r["dist_ma200_pct"]
        below200 = (r["ma200"] is not None and r["close"] < r["ma200"])
        ma50lt200 = (r["ma50"] is not None and r["ma200"] is not None and r["ma50"] < r["ma200"])

        if r.get("recently_lost_ma200"):
            alerts_risk.append(f"🟥 {t} pérdida reciente de MA200")
        if below200:
            alerts_risk.append(f"🚨 {t} bajo MA200")
        if d200 is not None and 0 < d200 <= NEAR_MA200_PCT:
            alerts_risk.append(f"⚠️ {t} cerca MA200 (+{d200:.2f}%)")
        if ma50lt200:
            alerts_risk.append(f"📉 {t} MA50<MA200")
        if r["breakout_3m"]:
            q = r.get("breakout_quality_pct")
            qtxt = f" ({q:+.1f}%)" if q is not None else ""
            alerts_momo.append(f"🚀 {t} ruptura 3M{qtxt}")

        if below200 and ma50lt200:
            reduce.append(t)
        elif d200 is not None and 0 <= d200 <= 5:
            watch.append(t)

        # Oportunidad candidata (base)
        base_opp = (
            r["breakout_3m"]
            and r["ma50_gt_ma200"]
            and (r["ma200"] is not None and r["close"] > r["ma200"])
        )

        if base_opp:
            # filtro A: no demasiado extendida sobre MA200
            if d200 is not None and 0 <= d200 <= OPP_MAX_EXT_MA200_PCT:
                opp_strict.append(r)
            else:
                opp_reject_ext.append(t)

    exposure = exposure_continuous(pct_over200, pct_ma50_over200, avg_score, len(alerts_risk), denom)
    semaforo = exposure_light(exposure)
    regime = regime_text(pct_over200, pct_ma50_over200, avg_score)

    # Movers vs última ejecución (score) (todo el universo)
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

    # Top/Bottom: cartera
    top = sorted(rows_cartera, key=lambda x: x["score"], reverse=True)[:6]
    bottom = sorted(rows_cartera, key=lambda x: x["score"])[:6]

    # Scouting: ideas
    scout_breakouts = [r for r in rows_scout if r["breakout_3m"]]
    scout_top = sorted(rows_scout, key=lambda x: x["score"], reverse=True)[:6]

    # ========================================================
    # BUILD REPORT (diseño)
    # ========================================================

    lines = []
    lines.append(f"📊 *INFORME DIARIO* — {today}")
    lines.append("")

    # Resumen rápido (dashboard)
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📦 *Resumen rápido*")
    lines.append(f"• Universo: *{len(rows)}* | Cartera: *{len(rows_cartera)}* | Scouting: *{len(rows_scout)}* | Sin datos: *{len(failed)}*")
    lines.append(f"• Régimen (cartera): *{regime}*")
    lines.append(f"• Exposición sugerida (cartera): *{semaforo} {exposure:.0f}%*")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    # Estructura de cartera
    lines.append("🧱 *Estructura de cartera*")
    lines.append(f"• Sobre MA200: *{pct_over200:.0f}%* ({len(over200)}/{denom})")
    lines.append(f"• MA50 > MA200: *{pct_ma50_over200:.0f}%* ({len(ma50_over200)}/{denom})")
    lines.append(f"• Verdes (score ≥70): *{n_green}/{denom}* ({pct_green:.0f}%) | Rojos (score <40): *{n_red}/{denom}* ({pct_red:.0f}%)")
    lines.append(f"• Score medio: *{avg_score:.0f}/100*")
    lines.append(f"• Bajo MA200: *{len(under200)}* | Cerca MA200 (≤{NEAR_MA200_PCT:.1f}%): *{len(near200)}* | Breakouts 3M: *{len(breakouts_cartera)}*")
    lines.append("")

    # E) movers del día (cartera)
    lines.append("⚡ *Movimientos del día (cartera, 1D)*")
    if top_1d:
        lines.append("• Top 1D: " + ", ".join([f"{r['ticker']} {_fmt_pct(r['ret_1d_pct'], 1, signed=True)}" for r in top_1d]))
    else:
        lines.append("• Top 1D: n/a")
    if bot_1d:
        lines.append("• Bottom 1D: " + ", ".join([f"{r['ticker']} {_fmt_pct(r['ret_1d_pct'], 1, signed=True)}" for r in bot_1d]))
    else:
        lines.append("• Bottom 1D: n/a")
    lines.append("")

    # Lectura estratégica
    lines.append("🧭 *Lectura estratégica (cartera)*")
    if len(under200) >= max(2, int(0.25 * denom)):
        lines.append("• Prioridad: *proteger* (daño estructural relevante).")
    elif len(near200) >= max(2, int(0.20 * denom)):
        lines.append("• Prioridad: *controlar riesgo* (varios en zona crítica).")
    else:
        lines.append("• Prioridad: *selectivo* (estructura razonable).")

    # Recuperación MA200
    if recover200:
        rec = ", ".join([r["ticker"] for r in sorted(recover200, key=lambda x: x["dist_ma200_pct"])[:10]])
        lines.append(f"• Recuperación MA200 (−{RECOVER_MA200_PCT:.0f}%..0%): {rec}")
    else:
        lines.append(f"• Recuperación MA200 (−{RECOVER_MA200_PCT:.0f}%..0%): n/a")

    # Riesgo por ATR
    if top_atr:
        atr_txt = ", ".join([f"{r['ticker']} {_fmt_pct(r['atr_pct'], 1, signed=False)}" for r in top_atr])
        lines.append(f"• Riesgo (ATR alto): {atr_txt}")
    else:
        lines.append("• Riesgo (ATR alto): n/a")

    # Pérdidas recientes MA200
    if recently_lost:
        lost_txt = ", ".join([r["ticker"] for r in recently_lost])
        lines.append(f"• Pérdida reciente MA200: {lost_txt}")
    else:
        lines.append("• Pérdida reciente MA200: n/a")

    lines.append("")

    # Acciones sugeridas
    lines.append("🛠️ *Acciones sugeridas (solo cartera)*")
    lines.append(f"• Reducir/recortar: {', '.join(sorted(set(reduce))) if reduce else 'n/a'}")
    lines.append(f"• Vigilar (0%..+5% MA200): {', '.join(sorted(set(watch))) if watch else 'n/a'}")
    lines.append(f"• Oportunidades (setup fuerte, no extendido): {', '.join([r['ticker'] for r in sorted(opp_strict, key=lambda x: x['score'], reverse=True)]) if opp_strict else 'n/a'}")
    if opp_reject_ext:
        # no lo hago muy largo; solo 8
        lines.append(f"• (Info) Señales descartadas por extensión >{OPP_MAX_EXT_MA200_PCT:.0f}%: {', '.join(sorted(set(opp_reject_ext))[:8])}")

    # Movers vs última ejecución (score)
    if movers_sorted:
        lines.append("")
        lines.append("🔁 *Movers vs última ejecución (score)*")
        for t, ds, dd in movers_sorted:
            sgn = "+" if ds >= 0 else ""
            dd_txt = f" | ΔvsMA200 {dd:+.2f}pp" if dd is not None else ""
            lines.append(f"• {arrow_delta(ds)} *{t}*: Δscore {sgn}{ds:.0f}{dd_txt}")

    # Top / Bottom con explicación
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🏆 *Top 6 (cartera)*")
    lines.append("_Qué significa: los 6 valores con mejor estructura técnica según el score (tendencia + momentum)._")
    lines.append("")
    if top:
        for r in top:
            lines.append(mini_card(r, include_size_hint=False))
            lines.append("")
    else:
        lines.append("n/a")
        lines.append("")

    lines.append("🧊 *Bottom 6 (cartera)*")
    lines.append("_Qué significa: los 6 valores con peor estructura; candidatos a vigilancia o reducción si tu tesis se debilita._")
    lines.append("")
    if bottom:
        for r in bottom:
            lines.append(mini_card(r, include_size_hint=False))
            lines.append("")
    else:
        lines.append("n/a")
        lines.append("")

    # C) Tamaño sugerido (ATR) aplicado SOLO a oportunidades estrictas
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📏 *Tamaño sugerido (solo para oportunidades)*")
    lines.append("_Regla simple por ATR: ≤3% grande | 3–6% medio | >6% pequeño._")
    lines.append("")
    if opp_strict:
        for r in sorted(opp_strict, key=lambda x: x["score"], reverse=True)[:8]:
            lines.append(mini_card(r, include_size_hint=True))
            lines.append("")
    else:
        lines.append("n/a")
        lines.append("")

    # Scouting
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🔎 *Scouting (ideas, no acciones)*")
    if scout_breakouts:
        tickers = ", ".join([r["ticker"] for r in sorted(scout_breakouts, key=lambda x: x["score"], reverse=True)[:10]])
        lines.append(f"• Rupturas 3M: {tickers}")
    else:
        lines.append("• Rupturas 3M: n/a")

    if scout_top:
        lines.append("• Top 6 scouting por score:")
        lines.append("")
        for r in scout_top:
            lines.append(mini_card(r, include_size_hint=False))
            lines.append("")
    else:
        lines.append("• Top 6 scouting por score: n/a")
        lines.append("")

    if failed:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("⚠️ *Sin datos / error*")
        lines.append(", ".join(failed))

    send_telegram("\n".join(lines).rstrip())

    # ========================================================
    # ALERTAS (mensaje separado + leyenda)
    # ========================================================

    alert_lines = []
    if alerts_risk:
        alert_lines.append("🚨 *ALERTAS (Riesgo / Control — cartera)*")
        alert_lines.append("_Leyenda: 🟥 pérdida reciente MA200 | 🚨 bajo MA200 | ⚠️ cerca MA200 | 📉 MA50<MA200_")
        alert_lines.append("")
        alert_lines.extend(alerts_risk)

    if alerts_momo:
        if alert_lines:
            alert_lines.append("")
        alert_lines.append("🚀 *ALERTAS (Momentum — cartera)*")
        alert_lines.append("_Leyenda: 🚀 ruptura 3M (entre paréntesis: calidad = % sobre el máximo previo)._")
        alert_lines.append("")
        alert_lines.extend(alerts_momo)

    if alert_lines:
        send_telegram("\n".join(alert_lines))


if __name__ == "__main__":
    main()
