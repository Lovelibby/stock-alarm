"""보유주식 매수·매도 타이밍 알람봇.

GitHub Actions에서 30분마다 실행:
  1) 실제 시세(일봉)를 받아 이동평균선·볼린저밴드·RSI·CCI 계산
  2) CNN 공포탐욕지수, 코스피/코스닥 등락, 종목 뉴스 확인
  3) 같은 방향 신호가 2개 이상 겹치면 텔레그램 알림 (같은 날 같은 신호는 1번만)
  4) docs/index.html 대시보드 갱신 (GitHub Pages로 휴대폰에서 보기)
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
STATE_PATH = ROOT / "state.json"
DOCS = ROOT / "docs"
KST = dt.timezone(dt.timedelta(hours=9))
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
FALLING = {"FALLING", "LOWER_LIMIT", "5", "4"}


# ---------------------------------------------------------------- 데이터 수집

def fetch_ohlcv(code: str) -> pd.DataFrame:
    """일봉 OHLCV (장중에는 오늘 봉이 진행 중 값으로 들어옴)."""
    import FinanceDataReader as fdr  # 실행 환경에서만 필요

    start = (dt.datetime.now(KST) - dt.timedelta(days=400)).strftime("%Y-%m-%d")
    df = fdr.DataReader(code, start)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    return df.dropna().astype(float)


def _num(x) -> float:
    return float(str(x).replace(",", "").replace("%", "").replace("+", ""))


def _signed_ratio(d: dict) -> float:
    """네이버 응답의 등락률에 하락이면 음수 부호를 붙임."""
    ratio = abs(_num(d["fluctuationsRatio"]))
    cmp = d.get("compareToPreviousPrice") or {}
    falling = str(cmp.get("name", "")).upper() in FALLING or str(cmp.get("code", "")) in FALLING
    if not cmp:  # 부호 정보가 없으면 원래 값 그대로
        return _num(d["fluctuationsRatio"])
    return -ratio if falling else ratio


def _naver(path: str) -> dict:
    import requests

    r = requests.get(f"https://m.stock.naver.com/api/{path}", headers=UA, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_index_change(symbol: str) -> float | None:
    """코스피·코스닥 실시간 등락률(네이버). 실패하면 일봉으로 계산."""
    code = {"KS11": "KOSPI", "KQ11": "KOSDAQ"}[symbol]
    try:
        return _signed_ratio(_naver(f"index/{code}/basic"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 지수 {code} 실시간 실패: {e}")
    try:
        import FinanceDataReader as fdr

        start = (dt.datetime.now(KST) - dt.timedelta(days=10)).strftime("%Y-%m-%d")
        c = fdr.DataReader(symbol, start)["Close"].dropna()
        print(f"[warn] 지수 {code}: 일봉 기준으로 대신 계산 (기준일 {c.index[-1]:%m-%d})")
        return float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 지수 {symbol} 실패: {e}")
        return None


def fetch_quote(code: str) -> dict | None:
    """종목 실시간 현재가·등락률(네이버)."""
    try:
        d = _naver(f"stock/{code}/basic")
        q = {"price": _num(d["closePrice"]), "chg": _signed_ratio(d)}
        traded = d.get("localTradedAt") or ""
        q["date"] = traded[:10] if re.match(r"\d{4}-\d{2}-\d{2}", traded) else None
        return q
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {code} 실시간 시세 실패: {e}")
        return None


def apply_quote(df: pd.DataFrame, q: dict | None) -> pd.DataFrame:
    """일봉 마지막 값을 실시간 현재가로 맞춤 (오늘 봉이 없으면 추가)."""
    if not q:
        return df
    df = df.copy()
    price = q["price"]
    last_day = df.index[-1].strftime("%Y-%m-%d")
    if q.get("date") and q["date"] > last_day:
        df.loc[pd.Timestamp(q["date"])] = [price, price, price, price, 0.0]
    elif q.get("date") is None or q["date"] == last_day:
        i = df.index[-1]
        df.loc[i, "Close"] = price
        df.loc[i, "High"] = max(df.loc[i, "High"], price)
        df.loc[i, "Low"] = min(df.loc[i, "Low"], price)
    return df


def fetch_fear_greed() -> dict | None:
    """CNN 공포탐욕지수 (0=극단적 공포, 100=극단적 탐욕)."""
    import requests

    today = dt.datetime.now(KST).strftime("%Y-%m-%d")
    for url in (
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}",
    ):
        try:
            r = requests.get(
                url,
                headers={**UA, "Referer": "https://edition.cnn.com/", "Origin": "https://edition.cnn.com"},
                timeout=15,
            )
            r.raise_for_status()
            fg = r.json()["fear_and_greed"]
            return {"score": round(float(fg["score"]), 1), "rating": fg.get("rating", "")}
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 공포탐욕지수 실패 ({url[-30:]}): {e}")
    return None


def fetch_news(name: str, limit: int = 5) -> list[dict]:
    """구글 뉴스 RSS에서 최근 2일 기사 제목."""
    import requests

    q = urllib.parse.quote(f"{name} 주가")
    url = f"https://news.google.com/rss/search?q={q}+when:2d&hl=ko&gl=KR&ceid=KR:ko"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        items = ET.fromstring(r.content).findall("./channel/item")[:limit]
        return [
            {"title": (i.findtext("title") or "").strip(), "link": i.findtext("link") or ""}
            for i in items
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 뉴스 {name} 실패: {e}")
        return []


# ---------------------------------------------------------------- 지표 계산

def compute_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    c, h, l = out["Close"], out["High"], out["Low"]

    for n in p["ma"]:
        out[f"MA{n}"] = c.rolling(n).mean()

    n, k = p["bollinger_period"], p["bollinger_k"]
    mid = c.rolling(n).mean()
    std = c.rolling(n).std(ddof=0)
    out["BB_MID"], out["BB_UP"], out["BB_LO"] = mid, mid + k * std, mid - k * std

    # RSI (Wilder 방식 — HTS와 같은 계산)
    n = p["rsi_period"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = (100 - 100 / (1 + rs)).where(loss != 0, 100.0)

    # CCI
    n = p["cci_period"]
    tp = (h + l + c) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out["CCI"] = (tp - sma) / (0.015 * mad)
    return out


# ---------------------------------------------------------------- 뉴스 점수

def news_score(headlines: list[dict], cfg: dict) -> int:
    """긍정 키워드 기사 수 - 부정 키워드 기사 수 → +1 / 0 / -1."""
    pos = neg = 0
    for h in headlines:
        t = h["title"]
        if any(w in t for w in cfg["negative_words"]):
            neg += 1
        elif any(w in t for w in cfg["positive_words"]):
            pos += 1
    if pos - neg >= 2:
        return 1
    if neg - pos >= 2:
        return -1
    return 0


# ---------------------------------------------------------------- 신호 판정

def evaluate(ind: pd.DataFrame, fg: dict | None, nscore: int) -> dict:
    t = CONFIG["thresholds"]
    last, prev = ind.iloc[-1], ind.iloc[-2]
    buy, sell = [], []

    rsi, cci, close = last["RSI"], last["CCI"], last["Close"]
    if rsi <= t["rsi_oversold"]:
        buy.append(f"RSI {rsi:.1f} 과매도")
    if rsi >= t["rsi_overbought"]:
        sell.append(f"RSI {rsi:.1f} 과매수")

    if cci <= t["cci_oversold"]:
        turning = " (반등 시작)" if cci > prev["CCI"] else ""
        buy.append(f"CCI {cci:.0f} 과매도{turning}")
    if cci >= t["cci_overbought"]:
        turning = " (꺾이는 중)" if cci < prev["CCI"] else ""
        sell.append(f"CCI {cci:.0f} 과매수{turning}")

    if close <= last["BB_LO"]:
        buy.append("볼린저 하단 터치")
    if close >= last["BB_UP"]:
        sell.append("볼린저 상단 터치")

    if prev["MA5"] <= prev["MA20"] and last["MA5"] > last["MA20"]:
        buy.append("5일선↗20일선 골든크로스")
    elif prev["Close"] < prev["MA20"] and close >= last["MA20"]:
        buy.append("20일선 회복")
    if prev["MA5"] >= prev["MA20"] and last["MA5"] < last["MA20"]:
        sell.append("5일선↘20일선 데드크로스")
    elif prev["Close"] > prev["MA20"] and close < last["MA20"]:
        sell.append("20일선 이탈")

    # 여기까지가 종목 자체 기술적 지표. 아래(시장 심리·뉴스)는 보조 가점이라
    # 기술적 지표가 하나도 없으면 신호를 내지 않음 (시장 공포만으로 전 종목 매수 신호가 뜨는 것 방지)
    tech_buy, tech_sell = len(buy), len(sell)

    if fg:
        if fg["score"] <= t["fear_extreme"]:
            buy.append(f"공포탐욕 {fg['score']:.0f} 극단적 공포")
        if fg["score"] >= t["greed_extreme"]:
            sell.append(f"공포탐욕 {fg['score']:.0f} 극단적 탐욕")

    if nscore > 0:
        buy.append("호재 뉴스 우세")
    if nscore < 0:
        sell.append("악재 뉴스 우세")

    s = CONFIG["signal"]

    def level(n: int) -> str | None:
        if n >= s["strong_score"]:
            return "strong"
        if n >= s["min_score"]:
            return "normal"
        return None

    return {
        "buy": buy,
        "sell": sell,
        "buy_level": level(len(buy)) if tech_buy else None,
        "sell_level": level(len(sell)) if tech_sell else None,
    }


# ---------------------------------------------------------------- 텔레그램

def send_telegram(text: str) -> bool:
    import requests

    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[info] 텔레그램 설정 없음 — 메시지 출력만:\n" + text)
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"},
        timeout=15,
    )
    if not r.ok:
        print(f"[warn] 텔레그램 실패: {r.status_code} {r.text[:200]}")
    return r.ok


def won(x: float) -> str:
    return f"{x:,.0f}원"


def build_alert(row: dict, side: str, fg: dict | None, idx: dict) -> str:
    sig = row["signal"]
    reasons = sig[side]
    strong = sig[f"{side}_level"] == "strong"
    label = "매수" if side == "buy" else "매도"
    icon = "🟢" if side == "buy" else "🔴"
    lines = [f"{icon}{' 🔥' if strong else ''} <b>{label} 신호 — {html.escape(row['name'])}</b>"]
    lines.append(f"현재가 {won(row['close'])} ({row['chg']:+.2f}%)")
    if row.get("qty"):
        lines.append(
            f"평단 {won(row['avg'])} · 수익률 {row['ret']:+.1f}% · 평가손익 {row['pnl']:+,.0f}원"
        )
    lines.append("근거: " + ", ".join(html.escape(r) for r in reasons))
    lines.append(
        f"RSI {row['rsi']:.1f} · CCI {row['cci']:.0f} · 볼린저 위치 {row['bb_pos']:.0f}%"
    )
    if row["news"]:
        lines.append("뉴스: " + html.escape(row["news"][0]["title"][:60]))
    mk = []
    if fg:
        mk.append(f"공포탐욕 {fg['score']:.0f}")
    for k, v in idx.items():
        if v is not None:
            mk.append(f"{k} {v:+.2f}%")
    if mk:
        lines.append("시장: " + " · ".join(mk))
    lines.append(f"<i>{row['asof']} 기준 · 참고용 기술적 신호이며 투자 판단은 본인 몫입니다.</i>")
    return "\n".join(lines)


# ---------------------------------------------------------------- 상태(중복 알림 방지)

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 대시보드

def sparkline(ind: pd.DataFrame, n: int = 60) -> str:
    d = ind.tail(n)
    vals = pd.concat([d["Close"], d["BB_UP"], d["BB_LO"]]).dropna()
    lo, hi = float(vals.min()), float(vals.max())
    w, h = 300, 70
    span = hi - lo or 1

    def pts(s: pd.Series) -> str:
        s = s.reset_index(drop=True)
        return " ".join(
            f"{i * w / (len(s) - 1):.1f},{h - (v - lo) / span * h:.1f}"
            for i, v in s.items()
            if not np.isnan(v)
        )

    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" class="spark" role="img" '
        f'aria-label="최근 {n}일 종가와 볼린저밴드">'
        f'<polyline class="bb" points="{pts(d["BB_UP"])}"/>'
        f'<polyline class="bb" points="{pts(d["BB_LO"])}"/>'
        f'<polyline class="ma" points="{pts(d["MA20"])}"/>'
        f'<polyline class="px" points="{pts(d["Close"])}"/></svg>'
    )


def render_dashboard(rows: list[dict], fg: dict | None, idx: dict, now: dt.datetime, show_pf: bool) -> str:
    e = html.escape

    def badge(r: dict) -> str:
        s = r["signal"]
        out = []
        if s["buy_level"]:
            out.append(f'<span class="b buy">{"🔥 " if s["buy_level"] == "strong" else ""}매수 신호</span>')
        if s["sell_level"]:
            out.append(f'<span class="b sell">{"🔥 " if s["sell_level"] == "strong" else ""}매도 신호</span>')
        return "".join(out) or '<span class="b none">관망</span>'

    def rsi_cls(v: float) -> str:
        return "lo" if v <= 30 else "hi" if v >= 70 else ""

    def cci_cls(v: float) -> str:
        return "lo" if v <= -100 else "hi" if v >= 100 else ""

    cards = []
    for r in sorted(rows, key=lambda r: -(len(r["signal"]["buy"]) + len(r["signal"]["sell"]))):
        if r.get("error"):
            cards.append(f'<div class="card"><h2>{e(r["name"])}</h2><p class="muted">데이터를 받지 못했어요: {e(r["error"])}</p></div>')
            continue
        reasons = [f'<li class="buy">{e(x)}</li>' for x in r["signal"]["buy"]] + [
            f'<li class="sell">{e(x)}</li>' for x in r["signal"]["sell"]
        ]
        pf = ""
        if show_pf and r.get("qty"):
            pf = (
                f'<div class="pf"><span>평단 {won(r["avg"])}</span>'
                f'<span class="{"up" if r["ret"] >= 0 else "down"}">{r["ret"]:+.1f}% ({r["pnl"]:+,.0f}원)</span></div>'
            )
        news = "".join(
            f'<li><a href="{e(n["link"])}" target="_blank" rel="noopener">{e(n["title"])}</a></li>'
            for n in r["news"][:3]
        )
        ma_state = " ".join(
            f'<span class="{"up" if r["close"] >= r[f"ma{n}"] else "down"}">MA{n}</span>'
            for n in CONFIG["indicators"]["ma"]
            if not np.isnan(r[f"ma{n}"])
        )
        cards.append(f"""
<div class="card">
  <div class="top"><h2>{e(r["name"])} <small>{e(r["code"])}</small></h2>{badge(r)}</div>
  <div class="price"><b>{r["close"]:,.0f}</b><span class="{"up" if r["chg"] >= 0 else "down"}">{r["chg"]:+.2f}%</span></div>
  {pf}
  {r["spark"]}
  <div class="grid">
    <div><label>RSI(14)</label><b class="{rsi_cls(r["rsi"])}">{r["rsi"]:.1f}</b></div>
    <div><label>CCI(14)</label><b class="{cci_cls(r["cci"])}">{r["cci"]:.0f}</b></div>
    <div><label>볼린저 위치</label><b>{r["bb_pos"]:.0f}%</b></div>
  </div>
  <div class="mas">종가 위/아래: {ma_state}</div>
  {f'<ul class="reasons">{"".join(reasons)}</ul>' if reasons else ""}
  {f'<details><summary>최근 뉴스</summary><ul class="news">{news}</ul></details>' if news else ""}
</div>""")

    fg_txt = f'{fg["score"]:.0f} <small>{e(fg["rating"])}</small>' if fg else "확인 불가"
    fg_cls = "" if not fg else ("lo" if fg["score"] <= 25 else "hi" if fg["score"] >= 75 else "")
    idx_html = "".join(
        f'<div><label>{e(k)}</label><b class="{"up" if (v or 0) >= 0 else "down"}">{"-" if v is None else f"{v:+.2f}%"}</b></div>'
        for k, v in idx.items()
    )
    total_html = ""
    if show_pf:
        held = [r for r in rows if r.get("qty") and not r.get("error")]
        if held:
            cost = sum(r["avg"] * r["qty"] for r in held)
            val = sum(r["close"] * r["qty"] for r in held)
            total_html = (
                f'<div><label>평가금액</label><b>{val:,.0f}</b></div>'
                f'<div><label>총 손익</label><b class="{"up" if val >= cost else "down"}">{(val / cost - 1) * 100:+.1f}%</b></div>'
            )
    n_sig = sum(1 for r in rows if not r.get("error") and (r["signal"]["buy_level"] or r["signal"]["sell_level"]))

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>내 주식 타이밍 보드</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1a1d21;--mut:#6b7280;--line:#e5e7eb;--up:#e5383b;--down:#2563eb;--buy:#0f9d58;--sell:#e5383b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--card:#181b21;--fg:#e8eaed;--mut:#9aa0a6;--line:#2a2e35}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}}
main{{max-width:720px;margin:0 auto;padding:16px}}h1{{font-size:20px;margin:4px 0}}.muted,small,label{{color:var(--mut)}}
.sum{{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin:12px 0}}
.sum div,.grid div{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}}
label{{display:block;font-size:12px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}}
.top{{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}}h2{{font-size:17px;margin:0}}
.price{{display:flex;gap:8px;align-items:baseline;margin:6px 0}}.price b{{font-size:24px}}
.up{{color:var(--up)}}.down{{color:var(--down)}}.lo{{color:var(--buy)}}.hi{{color:var(--sell)}}
.b{{font-size:13px;font-weight:600;padding:3px 9px;border-radius:99px;margin-left:4px}}.b.buy{{background:#0f9d5822;color:var(--buy)}}.b.sell{{background:#e5383b22;color:var(--sell)}}.b.none{{background:var(--line);color:var(--mut)}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}}.grid b{{font-size:17px}}
.spark{{width:100%;height:70px;margin-top:6px}}.spark polyline{{fill:none;stroke-width:1.5;vector-effect:non-scaling-stroke}}
.spark .px{{stroke:var(--fg)}}.spark .bb{{stroke:var(--mut);stroke-dasharray:3 3;opacity:.6}}.spark .ma{{stroke:#f59e0b}}
.mas{{font-size:13px;margin-top:8px}}.mas span{{margin-right:6px;font-weight:600}}
.reasons{{margin:8px 0 0;padding-left:18px;font-size:14px}}li.buy{{color:var(--buy)}}li.sell{{color:var(--sell)}}
.pf{{display:flex;justify-content:space-between;font-size:14px}}details{{margin-top:8px;font-size:14px}}.news a{{color:inherit}}
footer{{font-size:12px;color:var(--mut);margin:20px 0}}
</style></head><body><main>
<h1>내 주식 타이밍 보드</h1>
<div class="muted">{now:%Y-%m-%d %H:%M} 기준 · 신호 {n_sig}개 · 10분마다 새로고침</div>
<div class="sum">
  <div><label>공포탐욕(CNN)</label><b class="{fg_cls}">{fg_txt}</b></div>
  {idx_html}{total_html}
</div>
{"".join(cards)}
<footer>그래프: 검정=종가, 주황=20일선, 점선=볼린저밴드(20,2). 신호 기준: 같은 방향 지표가 2개 이상이면 신호, 3개 이상이면 🔥.
참고용 기술적 신호이며 투자 판단은 본인 몫입니다.</footer>
</main></body></html>"""


# ---------------------------------------------------------------- 실행

def load_portfolio() -> dict:
    raw = os.getenv("PORTFOLIO_JSON", "").strip()
    if not raw and (ROOT / "portfolio.json").exists():
        raw = (ROOT / "portfolio.json").read_text(encoding="utf-8")
    if not raw:
        return {}
    data = json.loads(raw)
    return {str(k).zfill(6): v for k, v in data.items()}


def analyze(stock: dict, df: pd.DataFrame, pf: dict, fg: dict | None, news: list[dict]) -> dict:
    ind = compute_indicators(df, CONFIG["indicators"])
    last, prev = ind.iloc[-1], ind.iloc[-2]
    sig = evaluate(ind, fg, news_score(news, CONFIG["news"]))
    width = last["BB_UP"] - last["BB_LO"]
    row = {
        **stock,
        "asof": ind.index[-1].strftime("%Y-%m-%d"),
        "close": float(last["Close"]),
        "chg": float((last["Close"] / prev["Close"] - 1) * 100),
        "rsi": float(last["RSI"]),
        "cci": float(last["CCI"]),
        "bb_pos": float((last["Close"] - last["BB_LO"]) / width * 100) if width else 50.0,
        "signal": sig,
        "news": news,
        "spark": sparkline(ind),
    }
    for n in CONFIG["indicators"]["ma"]:
        row[f"ma{n}"] = float(last[f"MA{n}"])
    h = pf.get(stock["code"])
    if h:
        qty, avg = float(h["qty"]), float(h["avg"])
        row.update(qty=qty, avg=avg, ret=(row["close"] / avg - 1) * 100, pnl=(row["close"] - avg) * qty)
    return row


def main() -> int:
    now = dt.datetime.now(KST)
    pf = load_portfolio()
    show_pf = os.getenv("SHOW_PORTFOLIO", "false").lower() == "true"
    fg = fetch_fear_greed()
    idx = {"코스피": fetch_index_change("KS11"), "코스닥": fetch_index_change("KQ11")}

    rows = []
    for s in CONFIG["stocks"]:
        try:
            df = fetch_ohlcv(s["code"])
            if len(df) < 130:
                raise ValueError(f"일봉 {len(df)}개뿐")
            q = fetch_quote(s["code"])
            row = analyze(s, apply_quote(df, q), pf, fg, fetch_news(s["name"]))
            if q:  # 등락률은 증권사 앱과 같은 공식 값 사용
                row["chg"] = q["chg"]
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {s['name']} 실패: {e}")
            rows.append({**s, "error": str(e)[:80]})

    # 알림: 같은 봉 날짜에 같은 신호는 1번만, 일반→🔥 강한 신호로 올라가면 다시 알림
    state = load_state()
    sent_any = 0
    for r in rows:
        if r.get("error"):
            continue
        for side in ("buy", "sell"):
            lvl = r["signal"][f"{side}_level"]
            if not lvl:
                continue
            key = f"{r['asof']}:{r['code']}:{side}"
            if state.get(key) == lvl or (state.get(key) == "strong" and lvl == "normal"):
                continue
            if send_telegram(build_alert(r, side, fg, idx)):
                sent_any += 1
            state[key] = lvl
    if state:  # 최근 7일치만 보관
        newest = dt.date.fromisoformat(max(k[:10] for k in state))
        cutoff = (newest - dt.timedelta(days=7)).isoformat()
        state = {k: v for k, v in state.items() if k[:10] >= cutoff}
    save_state(state)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(render_dashboard(rows, fg, idx, now, show_pf), encoding="utf-8")
    print(f"완료: {len(rows)}종목, 새 알림 {sent_any}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
