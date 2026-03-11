import os
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

EXCEL_PATH = os.environ.get(
    "CARTERA_XLSX_PATH",
    "/content/drive/MyDrive/Inversiones/Cartera.xlsx"
)

SHEET_CARTERA = "Cartera"
SHEET_CASH = "Cash"

TZ = ZoneInfo("Europe/Madrid")
MAX_TG_LEN = 3800

# Si algún ticker de tu Excel no coincide con Yahoo, lo mapeas aquí
TICKER_MAP = {
    # "UN": "UL",
}

# Buckets esperados
VALID_BUCKETS = {"Europa", "Growth", "HighRisk"}
VALID_CCY = {"EUR", "USD"}


# ============================================================
# TELEGRAM
# ============================================================

def _split_markdown_safe(text: str, max_len: int) -> list[str]:
    lines = (text or "").splitlines(True)
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
        raise RuntimeError("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chat_id = str(TELEGRAM_CHAT_ID).strip()

    for chunk in _split_markdown_safe(text, MAX_TG_LEN):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


# ============================================================
# HELPERS
# ============================================================

def as_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def to_float_or_none(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def fmt_pct(x, nd=1, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}%" if signed else f"{x:.{nd}f}%"


def fmt_eur(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# LOAD EXCEL
# ============================================================

def load_portfolio():
    cartera = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_CARTERA)
    cash = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_CASH)

    cartera.columns = [str(c).strip() for c in cartera.columns]
    cash.columns = [str(c).strip() for c in cash.columns]

    required = {"Ticker", "Cantidad", "PrecioMedio", "Divisa", "Bucket"}
    missing = required - set(cartera.columns)
    if missing:
        raise ValueError(f"Faltan columnas en hoja Cartera: {sorted(missing)}")

    cartera["Ticker"] = cartera["Ticker"].astype(str).str.strip()
    cartera["Divisa"] = cartera["Divisa"].astype(str).str.strip().str.upper()
    cartera["Bucket"] = cartera["Bucket"].astype(str).str.strip()

    bad_ccy = sorted(set(cartera["Divisa"]) - VALID_CCY)
    if bad_ccy:
        raise ValueError(f"Divisas no válidas en Cartera: {bad_ccy}")

    bad_bucket = sorted(set(cartera["Bucket"]) - VALID_BUCKETS)
    if bad_bucket:
        raise ValueError(f"Buckets no válidos en Cartera: {bad_bucket}")

    cartera["Ticker_YF"] = cartera["Ticker"].map(lambda x: TICKER_MAP.get(x, x))

    return cartera, cash


def get_cash_values(cash_df: pd.DataFrame):
    col_key = None
    col_val = None

    for c in cash_df.columns:
        lc = c.lower()
        if lc in {"concepto", "concept", "item", "tipo", "name"}:
            col_key = c
        if lc in {"importe", "amount", "valor", "value"}:
            col_val = c

    if col_key is None or col_val is None:
        return 0.0, 0.0

    def get_cash(label, default=0.0):
        m = cash_df[cash_df[col_key].astype(str).str.strip().str.lower() == label.lower()]
        if m.empty:
            return float(default)
        return float(m[col_val].iloc[0])

    cash_eur = get_cash("CashEUR", 0.0)
    cash_usd = get_cash("CashUSD", 0.0)
    return cash_eur, cash_usd


# ============================================================
# MARKET DATA
# ============================================================

def get_fx_eurusd():
    fx = yf.download("EURUSD=X", period="10d", interval="1d", auto_adjust=False, progress=False)
    if fx is None or fx.empty:
        raise RuntimeError("No se pudo descargar EURUSD=X")
    close = as_series(fx["Close"]).dropna()
    return float(close.iloc[-1])


def get_history(ticker: str, period="420d"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None

    close = as_series(df["Close"]).dropna()
    high = as_series(df["High"]).dropna() if "High" in df else None
    vol = as_series(df["Volume"]).dropna() if "Volume" in df else None

    if close is None or close.empty:
        return None

    return {
        "raw": df,
        "close": close,
        "high": high,
        "volume": vol,
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_position(row, eurusd_rate):
    ticker = row["Ticker"]
    ticker_yf = row["Ticker_YF"]
    qty = float(row["Cantidad"])
    avg = float(row["PrecioMedio"])
    divisa = row["Divisa"]
    bucket = row["Bucket"]

    hist = get_history(ticker_yf, period="420d")
    if hist is None:
        return None

    close = hist["close"]
    high = hist["high"]
    vol = hist["volume"]

    if len(close) < 220:
        return None

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else None

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])

    vol_rel = None
    vol_2x = False
    if vol is not None and len(vol) >= 30:
        vol_avg30 = float(vol.rolling(30).mean().iloc[-1])
        last_vol = float(vol.iloc[-1])
        if vol_avg30 != 0:
            vol_rel = last_vol / vol_avg30
            vol_2x = vol_rel >= 2.0

    ret_1d = None
    if prev not in (None, 0):
        ret_1d = (last / prev - 1.0) * 100.0

    rupture_20d = False
    if high is not None and len(high) >= 21:
        max_prev_20 = float(high.rolling(20).max().iloc[-2])
        rupture_20d = last > max_prev_20

    # YTD aproximado para la posición actual (sin flujos)
    year = datetime.now(TZ).year
    close_ytd = close[close.index.year == year]
    ytd_pct = None
    if len(close_ytd) >= 2:
        first_ytd = float(close_ytd.iloc[0])
        if first_ytd != 0:
            ytd_pct = (last / first_ytd - 1.0) * 100.0

    lose_ma50 = last < ma50
    lose_ma200 = last < ma200

    # Conversión EUR
    last_eur = last / eurusd_rate if divisa == "USD" else last
    prev_eur = prev / eurusd_rate if (divisa == "USD" and prev is not None) else prev
    avg_eur = avg / eurusd_rate if divisa == "USD" else avg

    value_eur = last_eur * qty
    cost_eur = avg_eur * qty
    pnl_eur = value_eur - cost_eur
    pnl_pct = (last_eur / avg_eur - 1.0) * 100.0 if avg_eur not in (None, 0) else None

    return {
        "Ticker": ticker,
        "Ticker_YF": ticker_yf,
        "Bucket": bucket,
        "Divisa": divisa,
        "Cantidad": qty,
        "PrecioHoy": last,
        "PrecioAyer": prev,
        "%Dia": ret_1d,
        "YTD_%": ytd_pct,
        "PrecioMedio": avg,
        "PrecioHoy_EUR": last_eur,
        "Valor_EUR": value_eur,
        "Coste_EUR": cost_eur,
        "PnL_EUR": pnl_eur,
        "PnL_%": pnl_pct,
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
        "RSI14": rsi14,
        "VolRel": vol_rel,
        "Ruptura_20D": rupture_20d,
        "Pierde_MA50": lose_ma50,
        "Pierde_MA200": lose_ma200,
        "Vol_>2x": vol_2x,
        "Alerta_%Dia_±7": (ret_1d is not None and abs(ret_1d) >= 7),
    }


# ============================================================
# REPORT
# ============================================================

def build_report(df: pd.DataFrame, cash_total_eur: float, eurusd_rate: float):
    valor_total_eur = df["Valor_EUR"].sum() + cash_total_eur

    df = df.sort_values("Valor_EUR", ascending=False).copy()
    df["Peso_%"] = df["Valor_EUR"] / valor_total_eur * 100

    # Exposición divisa
    expo_usd_eur = df[df["Divisa"] == "USD"]["Valor_EUR"].sum()
    expo_usd_pct = expo_usd_eur / valor_total_eur * 100 if valor_total_eur else 0.0

    # Buckets
    bucket_pct = (df.groupby("Bucket")["Valor_EUR"].sum() / valor_total_eur * 100).sort_values(ascending=False)

    # Concentración
    mayor_pos = float(df.iloc[0]["Peso_%"]) if len(df) else 0.0
    top5 = float(df.head(5)["Peso_%"].sum())
    top10 = float(df.head(10)["Peso_%"].sum())

    # Variación diaria aprox ponderada
    df_var = df[df["%Dia"].notna()].copy()
    var_total_dia_pct = (
        (df_var["Valor_EUR"] * (df_var["%Dia"] / 100.0)).sum() / df_var["Valor_EUR"].sum() * 100.0
        if len(df_var) else 0.0
    )

    # YTD aprox cartera actual (sin flujos)
    df_ytd = df[df["YTD_%"].notna()].copy()
    ytd_total_pct = (
        (df_ytd["Valor_EUR"] * (df_ytd["YTD_%"] / 100.0)).sum() / df_ytd["Valor_EUR"].sum() * 100.0
        if len(df_ytd) else None
    )

    # Movers
    top_gan = df.sort_values("%Dia", ascending=False).head(6)
    top_per = df.sort_values("%Dia", ascending=True).head(6)

    # Alertas
    alertas = df[
        (df["Ruptura_20D"] == True) |
        (df["Pierde_MA50"] == True) |
        (df["Pierde_MA200"] == True) |
        (df["Alerta_%Dia_±7"] == True) |
        (df["Vol_>2x"] == True)
    ].copy()

    # Candidatos añadir en fortaleza
    candidatos_add = df[
        (df["PrecioHoy"] > df["MA200"]) &
        (df["RSI14"] >= 55) &
        (df["VolRel"].fillna(0) >= 1.2)
    ].sort_values(
        ["Ruptura_20D", "VolRel", "RSI14", "Peso_%"],
        ascending=[False, False, False, False]
    ).head(10)

    now_txt = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"📊 *INFORME CARTERA REAL* — {now_txt}")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🔎 *Resumen ejecutivo*")
    lines.append(f"• Valor total: *{fmt_eur(valor_total_eur, 0)}*")
    lines.append(f"• Cash total: {fmt_eur(cash_total_eur, 0)}")
    lines.append(f"• Variación diaria (aprox): *{fmt_pct(var_total_dia_pct, 2)}*")
    lines.append(f"• YTD cartera actual (sin flujos): *{fmt_pct(ytd_total_pct, 1)}*")
    lines.append(f"• Exposición USD: *{fmt_pct(expo_usd_pct, 1, signed=False)}*")
    lines.append(f"• Mayor posición: *{fmt_pct(mayor_pos, 2, signed=False)}*")
    lines.append(f"• Top 5: *{fmt_pct(top5, 2, signed=False)}* | Top 10: *{fmt_pct(top10, 2, signed=False)}*")
    lines.append("")

    lines.append("🎯 *Buckets*")
    for bucket, pct in bucket_pct.items():
        lines.append(f"• {bucket}: {fmt_pct(float(pct), 1, signed=False)}")
    lines.append("")

    lines.append("📈 *Top ganadoras del día*")
    for _, r in top_gan.iterrows():
        lines.append(f"• {r['Ticker']}: {fmt_pct(r['%Dia'], 2)} | peso {fmt_pct(r['Peso_%'], 1, signed=False)} | {r['Bucket']}")
    lines.append("")

    lines.append("📉 *Top perdedoras del día*")
    for _, r in top_per.iterrows():
        lines.append(f"• {r['Ticker']}: {fmt_pct(r['%Dia'], 2)} | peso {fmt_pct(r['Peso_%'], 1, signed=False)} | {r['Bucket']}")
    lines.append("")

    lines.append("🧪 *Snapshot técnico por ticker*")
    for _, r in df.iterrows():
        tags = []
        if r["Ruptura_20D"]:
            tags.append("Breakout")
        if r["Pierde_MA50"]:
            tags.append("↓MA50")
        if r["Pierde_MA200"]:
            tags.append("↓MA200")
        if r["Vol_>2x"]:
            tags.append("Vol>2x")

        tag_txt = " | " + ", ".join(tags) if tags else ""
        lines.append(
            f"• *{r['Ticker']}* | RSI {r['RSI14']:.0f} | VolRel {r['VolRel']:.2f if pd.notna(r['VolRel']) else 'n/a'} | "
            f"MA20/50/200 {r['MA20']:.2f}/{r['MA50']:.2f}/{r['MA200']:.2f}{tag_txt}"
        )
    lines.append("")

    lines.append("🚨 *Alertas del día*")
    if alertas.empty:
        lines.append("• Sin alertas relevantes")
    else:
        for _, r in alertas.iterrows():
            tags = []
            if r["Ruptura_20D"]:
                tags.append("RUPTURA")
            if r["Pierde_MA50"]:
                tags.append("↓MA50")
            if r["Pierde_MA200"]:
                tags.append("↓MA200")
            if r["Alerta_%Dia_±7"]:
                tags.append("±7%")
            if r["Vol_>2x"]:
                tags.append("VOL>2x")
            lines.append(f"• {r['Ticker']} {fmt_pct(r['%Dia'], 2)} | {' / '.join(tags)}")
    lines.append("")

    lines.append("💡 *Acciones sugeridas (0–3) — Añadir en fortaleza*")
    if candidatos_add.empty:
        lines.append("• No hay setups claros hoy")
    else:
        for _, r in candidatos_add.head(3).iterrows():
            extra = " + ruptura" if r["Ruptura_20D"] else ""
            lines.append(
                f"• {r['Ticker']} ({r['Bucket']}) | RSI {r['RSI14']:.0f} | VolRel {r['VolRel']:.2f}{extra}"
            )

    return "\n".join(lines), df


# ============================================================
# MAIN
# ============================================================

def main():
    cartera, cash = load_portfolio()
    cash_eur, cash_usd = get_cash_values(cash)
    eurusd_rate = get_fx_eurusd()
    cash_total_eur = cash_eur + (cash_usd / eurusd_rate)

    rows = []
    failed = []

    for _, row in cartera.iterrows():
        r = analyze_position(row, eurusd_rate)
        if r is None:
            failed.append(str(row["Ticker"]))
        else:
            rows.append(r)

    if not rows:
        raise RuntimeError("No se pudo analizar ninguna posición del Excel.")

    df = pd.DataFrame(rows)
    report_text, df_final = build_report(df, cash_total_eur, eurusd_rate)

    if failed:
        report_text += "\n\n⚠️ *Sin datos / error*\n• " + ", ".join(failed)

    send_telegram(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
