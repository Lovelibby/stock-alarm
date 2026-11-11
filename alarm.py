"""국장·미장·코인 매수·매도 타이밍 알람봇.

GitHub Actions에서 주기적으로 실행:
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

import scoring

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

def fetch_ohlcv(code: str, source: str = "fdr") -> pd.DataFrame:
    """일봉 OHLCV. 국장·미장은 FinanceDataReader, 코인은 업비트(원화)."""
    if source == "upbit":
        return fetch_upbit(code)
    import FinanceDataReader as fdr  # 실행 환경에서만 필요

    start = (dt.datetime.now(KST) - dt.timedelta(days=620)).strftime("%Y-%m-%d")
    df = fdr.DataReader(code, start)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    return df.dropna().astype(float)


def fetch_upbit(market: str, total: int = 480) -> pd.DataFrame:
    """업비트 원화 마켓 일봉 (하루 = 매일 오전 9시 시작). 현재 진행 중인 봉 포함. 200개씩 나눠 받음."""
    import time

    import requests

    rows, to = [], None
    while len(rows) < total:
        params = {"market": market, "count": min(200, total - len(rows))}
        if to:
            params["to"] = to
        r = requests.get("https://api.upbit.com/v1/candles/days", params=params,
                         headers={"accept": "application/json"}, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        to = batch[-1]["candle_date_time_utc"]
        time.sleep(0.15)  # 업비트 요청 제한 배려
    df = pd.DataFrame(rows).drop_duplicates("candle_date_time_kst")
    df.index = pd.to_datetime(df["candle_date_time_kst"].str[:10])
    df = df.rename(columns={"opening_price": "Open", "high_price": "High", "low_price": "Low",
                            "trade_price": "Close", "candle_acc_trade_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index().astype(float)


def fetch_upbit_intraday(market: str, unit: int = 15, count: int = 96) -> pd.DataFrame | None:
    """업비트 분봉 (기본 15분봉 96개 = 최근 24시간)."""
    import requests

    try:
        r = requests.get(f"https://api.upbit.com/v1/candles/minutes/{unit}",
                         params={"market": market, "count": count},
                         headers={"accept": "application/json"}, timeout=15)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df.index = pd.to_datetime(df["candle_date_time_kst"])
        df = df.rename(columns={"opening_price": "Open", "high_price": "High", "low_price": "Low",
                                "trade_price": "Close", "candle_acc_trade_volume": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]].sort_index().astype(float)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {market} 분봉 실패: {e}")
        return None


def fetch_crypto_fear_greed() -> dict | None:
    """코인 공포탐욕지수 (alternative.me, 0~100)."""
    import requests

    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", headers=UA, timeout=15)
        r.raise_for_status()
        d = r.json()["data"][0]
        return {"score": float(d["value"]), "rating": d.get("value_classification", "")}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 코인 공포탐욕지수 실패: {e}")
        return None


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
    """지수 등락률. 코스피·코스닥은 네이버 실시간, 그 외(미국 지수)는 일봉으로 계산."""
    code = {"KS11": "KOSPI", "KQ11": "KOSDAQ"}.get(symbol, symbol)
    if code != symbol:
        try:
            return _signed_ratio(_naver(f"index/{code}/basic"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 지수 {code} 실시간 실패: {e}")
    try:
        import FinanceDataReader as fdr

        start = (dt.datetime.now(KST) - dt.timedelta(days=10)).strftime("%Y-%m-%d")
        c = fdr.DataReader(symbol, start)["Close"].dropna()
        if code != symbol:
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
        if d.get("accumulatedTradingVolume") not in (None, ""):
            q["volume"] = _num(d["accumulatedTradingVolume"])
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
        df.loc[pd.Timestamp(q["date"])] = [price, price, price, price, q.get("volume", 0.0)]
    elif q.get("date") is None or q["date"] == last_day:
        i = df.index[-1]
        df.loc[i, "Close"] = price
        df.loc[i, "High"] = max(df.loc[i, "High"], price)
        df.loc[i, "Low"] = min(df.loc[i, "Low"], price)
        if q.get("volume"):
            df.loc[i, "Volume"] = max(df.loc[i, "Volume"], q["volume"])
    return df


def fetch_investor(code: str, days: int = 5) -> list[dict]:
    """외국인·기관 일별 순매수 수량(주), 최근일이 앞. 장 마감 후 집계라 보통 전일까지 반영."""
    import io

    import requests

    try:
        data = _naver(f"stock/{code}/trend?pageSize={days}")
        rows = data if isinstance(data, list) else data.get("trends") or data.get("items") or []
        out = []
        for d in rows[:days]:
            out.append({
                "date": str(d.get("bizdate", ""))[:8],
                "foreign": _num(d["foreignerPureBuyQuant"]),
                "inst": _num(d["organPureBuyQuant"]),
            })
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {code} 수급(모바일) 실패: {e}")
    try:
        r = requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}", headers=UA, timeout=15)
        r.encoding = "euc-kr"
        for t in pd.read_html(io.StringIO(r.text)):
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            fi = next((i for i, c in enumerate(cols) if "외국인" in c and "순매매" in c), None)
            ii = next((i for i, c in enumerate(cols) if "기관" in c and "순매매" in c), None)
            if fi is None or ii is None:
                continue
            t = t.dropna(subset=[t.columns[fi], t.columns[ii]])
            out = [
                {"date": str(row.iloc[0]), "foreign": _num(row.iloc[fi]), "inst": _num(row.iloc[ii])}
                for _, row in t.head(days).iterrows()
            ]
            if out:
                return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {code} 수급(웹) 실패: {e}")
    return []


def market_today(now: dt.datetime, mid: str) -> dt.date:
    """시장 기준 '오늘' 날짜 (미장은 뉴욕 날짜, 코인은 업비트 일봉 기준 오전 9시 시작)."""
    if mid == "us":
        from zoneinfo import ZoneInfo

        return now.astimezone(ZoneInfo("America/New_York")).date()
    if mid == "coin":
        return (now - dt.timedelta(hours=9)).date()
    return now.date()


def session_elapsed(now: dt.datetime, mid: str = "kr") -> float:
    """오늘 거래가 얼마나 진행됐는지(0.25~1). 진행 중인 거래량을 하루치로 환산할 때 사용."""
    if mid == "coin":  # 24시간, 오전 9시(KST) 기준
        hrs = ((now.hour - 9) % 24) + now.minute / 60
        return max(hrs / 24, 0.25)
    if mid == "us":
        from zoneinfo import ZoneInfo

        now = now.astimezone(ZoneInfo("America/New_York"))
        open_min = 9 * 60 + 30
    else:
        open_min = 9 * 60
    if now.weekday() >= 5:
        return 1.0
    mins = (now.hour * 60 + now.minute) - open_min
    if mins <= 0 or mins >= 390:
        return 1.0
    return max(mins / 390, 0.25)  # 장 초반은 거래가 몰려 과대평가되므로 하한 0.25


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


def fetch_news(name: str, limit: int = 5, suffix: str = "주가") -> list[dict]:
    """구글 뉴스 RSS에서 최근 2일 기사 제목."""
    import requests

    q = urllib.parse.quote(f"{name} {suffix}".strip())
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
    out["VOL20"] = out["Volume"].rolling(20).mean().shift(1)
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


# ---------------------------------------------------------------- 신호 (점수 기반)

def signal_from_scores(timing: dict, risk: dict, held: bool) -> dict:
    """매수: 타이밍 75점 이상(85 이상 🔥). 매도: 보유 종목만, 매도위험 65 이상(80 이상 🐵)."""
    tv, rv = timing["value"], risk["value"]
    buy = "strong" if tv is not None and tv >= 85 else "normal" if tv is not None and tv >= 75 else None
    sell = None
    if held and rv is not None:
        sell = "strong" if rv >= 80 else "normal" if rv >= 65 else None
    return {"buy_level": buy, "sell_level": sell}


def summary_line(r: dict) -> str:
    t, k, sig = r["timing"], r["risk"], r["signal"]
    tv, kv = t["value"], k["value"]
    if tv is None:
        return "데이터가 부족해 판단을 보류해요."
    if sig["sell_level"] == "strong":
        return f"🐵 매도 위험 {kv:.0f}점으로 강한 경고예요. 수익 중이면 상당 부분 이익 실현을 검토해 보세요."
    if sig["sell_level"]:
        return f"매도 위험 {kv:.0f}점으로 주의 구간이에요. 일부 이익 실현이나 비중 축소를 검토해 보세요."
    if sig["buy_level"] == "strong":
        return f"🔥 매수 타이밍 {tv:.0f}점으로 차트상 매우 좋은 진입 구간이에요. 분할매수 구간을 참고하세요."
    if sig["buy_level"]:
        return f"매수 타이밍 {tv:.0f}점으로 좋은 편이에요. 한 번에 사기보다 분할매수 구간에서 나눠 사는 게 안전해요."
    if kv is not None and kv >= 45:
        return f"매수 타이밍 {tv:.0f}점, 매도 위험 {kv:.0f}점이에요. 추세가 약해지는 중이라 지켜보는 게 좋아요."
    return f"매수 타이밍 {tv:.0f}점({t['grade']}), 매도 위험 {kv:.0f}점({k['grade']})으로 특별한 신호는 없어요."


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


def money(x: float, cur: str = "KRW") -> str:
    if cur == "USD":
        return f"${x:,.2f}"
    return f"{x:,.0f}원" if abs(x) >= 100 else f"{x:,.2f}원"


def price_txt(x: float, cur: str = "KRW") -> str:
    if cur == "USD":
        return f"${x:,.2f}"
    return f"{x:,.0f}" if x >= 100 else f"{x:,.2f}"


def man(x: float, cur: str = "KRW") -> str:
    """차트 라벨용 짧은 가격: 229,500 → 22.9만, 1.5억, $123"""
    if cur == "USD":
        return f"${x:,.0f}" if x >= 100 else f"${x:.1f}"
    if x >= 1e8:
        return f"{x / 1e8:.2f}억"
    return f"{x / 10000:.1f}만" if x >= 10000 else f"{x:,.0f}"


def build_alert(row: dict, side: str, fg: dict | None, idx: dict) -> str:
    esc = lambda x: html.escape(str(x), quote=False)  # noqa: E731
    cur = row.get("cur", "KRW")
    sig, plan = row["signal"], row["plan"]
    strong = sig[f"{side}_level"] == "strong"
    head = ("🟢" + (" 🔥" if strong else "") + f" <b>[{esc(row['market'])}] {'강한 ' if strong else ''}매수 타이밍")\
        if side == "buy" else ("🔴" + (" 🐵" if strong else "") + f" <b>[{esc(row['market'])}] {'강한 ' if strong else ''}매도 경고")
    lines = [f"{head} — {esc(row['name'])}</b>", f"현재가 {money(row['close'], cur)} ({row['chg']:+.2f}%)"]
    if row.get("qty"):
        lines.append(f"평단 {money(row['avg'], cur)} · 수익률 {row['ret']:+.1f}% · 평가손익 {money(row['pnl'], cur)}")
    sc = row["timing"] if side == "buy" else row["risk"]
    lines.append(f"<b>{'매수 타이밍' if side == 'buy' else '매도 위험'} {sc['value']:.0f}/{sc['scale']}점 ({esc(sc['grade'])})</b>"
                 f" · 신뢰도 {sc['confidence']}%")
    top = sorted([i for i in sc["items"] if i["score"] is not None],
                 key=lambda i: -(i["score"] / i["max"]) if side == "buy" else -(i["score"] / i["max"]))[:3]
    lines.append("💡 이유")
    lines += [f"• {esc(i['name'])} {i['score']:.0f}/{i['max']}: {esc(i['reason'])}" for i in top]
    if side == "buy":
        if plan.get("buy_zone"):
            lo, hi = plan["buy_zone"]
            lines.append(f"📍 분할매수 구간 {money(lo, cur)}~{money(hi, cur)}")
        lines.append(f"🛑 손절 고려 {money(plan['stop'], cur)} · 🎯 1차 {money(plan['target1'], cur)} / 2차 {money(plan['target2'], cur)}")
    else:
        lines.append(f"🎯 1차 목표 {money(plan['target1'], cur)} · 🛑 손절 고려 {money(plan['stop'], cur)}")
    lines.append(esc(summary_line(row)))
    mk = []
    if fg:
        mk.append(f"{fg.get('label', '공포탐욕')} {fg['score']:.0f}")
    mk += [f"{k} {v:+.2f}%" for k, v in idx.items() if v is not None]
    if mk:
        lines.append("시장: " + " · ".join(mk))
    lines.append(f"<i>{row['asof']} 기준 · 차트 기반 참고 분석이며 투자 판단은 본인 몫입니다.</i>")
    return "\n".join(lines)


def build_price_alert(row: dict, kind: str) -> str:
    cur = row.get("cur", "KRW")
    name = html.escape(row["name"], quote=False)
    if kind == "target":
        return (f"🎯 <b>[{row['market']}] 목표가 도달 — {name}</b>\n현재가 {money(row['close'], cur)} ≥ 목표가 "
                f"{money(row['target'], cur)}\n수익률 {row['ret']:+.1f}% · 일부 이익 실현을 검토해 보세요.\n"
                f"<i>단순 가격 도달 알림이에요. 매도 위험 {row['risk']['value']:.0f}점({row['risk']['grade']})도 함께 보세요.</i>")
    return (f"🛑 <b>[{row['market']}] 손절가 도달 — {name}</b>\n현재가 {money(row['close'], cur)} ≤ 손절가 "
            f"{money(row['stop'], cur)}\n수익률 {row['ret']:+.1f}% · 미리 정한 기준에 닿았어요.\n"
            f"<i>단순 가격 도달 알림이에요. 투자 판단은 본인 몫입니다.</i>")


# ---------------------------------------------------------------- 상태(중복 알림 방지)

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 대시보드: 차트

def chart(ind: pd.DataFrame, marks: list[dict], n: int, cur: str, style: str, lv: dict,
          label_all: bool = True) -> str:
    """가격(봉/선) + 20·60·120일선 + 지지·저항선 + 매매 표시, 아래에 거래량·RSI·MACD."""
    start = max(len(ind) - n, 0)
    d = ind.iloc[start:]
    w, ph, pad = 320, 160, 22          # 가격 영역
    vh, rh, mh, gap = 34, 38, 38, 8      # 거래량·RSI·MACD 영역
    H = ph + gap + vh + gap + rh + gap + mh
    cols = ["High", "Low", "MA20"] if style == "candle" else ["Close", "BB_UP", "BB_LO", "MA20", "MA60", "MA120"]
    vals = d[cols].stack().dropna()
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1
    step = w / max(len(d) - 1, 1) if style == "line" else w / max(len(d), 1)
    off = 0 if style == "line" else step / 2

    def x(i: int) -> float:
        return off + i * step

    def y(v: float) -> float:
        return pad + (hi - v) / span * (ph - 2 * pad)

    def line(col: str, cls: str, yf=y) -> str:
        pts = " ".join(f"{x(i):.1f},{yf(v):.1f}" for i, v in enumerate(d[col]) if not np.isnan(v))
        return f'<polyline class="{cls}" points="{pts}"/>' if pts else ""

    svg = []
    if style == "line":
        svg += [line("BB_UP", "bb"), line("BB_LO", "bb")]
    svg += [line("MA120", "ma120"), line("MA60", "ma60"), line("MA20", "ma20")]
    # 지지·저항선 (현재 기준)
    used: list[float] = []
    for v, cls, tag in [(s, "sup", "지지") for s in lv["support"]] + [(r, "res", "저항") for r in lv["resistance"]]:
        if lo <= v <= hi:
            ty = y(v) - 2
            while any(abs(ty - u) < 9 for u in used):  # 라벨끼리 겹치면 살짝 비켜 씀
                ty += 9
            used.append(ty)
            svg.append(f'<line class="{cls}" x1="0" x2="{w}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                       f'<text class="lvl {cls}t" x="2" y="{ty:.1f}">{tag} {man(v, cur)}</text>')
    if style == "candle":
        bw = max(step * 0.6, 1.5)
        for i, (_, r) in enumerate(d.iterrows()):
            c = "cu" if r["Close"] >= r["Open"] else "cd"
            top, bot = y(max(r["Open"], r["Close"])), y(min(r["Open"], r["Close"]))
            svg.append(f'<line class="{c}" x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(r["High"]):.1f}" y2="{y(r["Low"]):.1f}"/>'
                       f'<rect class="{c}" x="{x(i) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(bot - top, 0.8):.1f}"/>')
    else:
        svg.append(line("Close", "px"))
    # 매매 표시
    for m in marks:
        if m["i"] < start:
            continue
        i = m["i"] - start
        px = float(d["Close"].iloc[i])
        cx, up = x(i), m["kind"] == "buy"
        ref = float(d["Low"].iloc[i] if up else d["High"].iloc[i]) if style == "candle" else px
        yy = min(y(ref) + 12, ph - 12) if up else max(y(ref) - 12, 12)
        title = "<title>" + html.escape(f'{m["day"]} {money(m["price"], cur)} · {m["label"]}') + "</title>"
        if m.get("strong"):
            svg.append(f'<text class="emo" x="{cx:.1f}" y="{yy + 5:.1f}" text-anchor="middle">{"🔥" if up else "🐵"}{title}</text>')
        elif up:
            svg.append(f'<path class="mk-buy" d="M{cx:.1f},{yy - 6:.1f} l-5,8 h10 z">{title}</path>')
        else:
            svg.append(f'<path class="mk-sell" d="M{cx:.1f},{yy + 6:.1f} l-5,-8 h10 z">{title}</path>')
        if label_all or m.get("strong"):
            tx = min(max(cx, 14), w - 14)
            svg.append(f'<text class="pl {"pb" if up else "ps"}" x="{tx:.1f}" y="{(yy + 16) if up else (yy - 10):.1f}" '
                       f'text-anchor="middle">{man(px, cur)}</text>')
    # 거래량
    y0 = ph + gap
    vmax = float(d["Volume"].max()) or 1
    bw = max(step * 0.6, 1)
    svg.append(f'<text class="pane-t" x="2" y="{y0 + 8}">거래량</text>')
    # 막대를 하나하나 rect로 그리면 파일이 커져서, 색깔별로 path 하나에 모아서 그림
    segs = {"vu": [], "vd": []}
    for i, (o_, c_, v_) in enumerate(zip(d["Open"], d["Close"], d["Volume"])):
        hgt = v_ / vmax * (vh - 4)
        segs["vu" if c_ >= o_ else "vd"].append(f"M{x(i):.1f} {y0 + vh:.1f}v{-hgt:.1f}")
    svg += [f'<path class="{k}b" style="stroke-width:{bw:.1f}" d="{"".join(v)}"/>' for k, v in segs.items() if v]
    # RSI
    y1 = y0 + vh + gap

    def yr(v: float) -> float:
        return y1 + (100 - v) / 100 * rh

    svg.append(f'<rect class="zone" x="0" y="{yr(100):.1f}" width="{w}" height="{yr(70) - yr(100):.1f}"/>'
               f'<rect class="zone" x="0" y="{yr(30):.1f}" width="{w}" height="{yr(0) - yr(30):.1f}"/>'
               f'<text class="pane-t" x="2" y="{y1 + 8}">RSI {d["RSI"].iloc[-1]:.0f}</text>')
    svg.append(line("RSI", "rsi", yr))
    # MACD (막대 = MACD - 시그널)
    y2 = y1 + rh + gap
    hist = d["MACD_HIST"].fillna(0)
    hm = float(hist.abs().max()) or 1
    mid = y2 + mh / 2
    svg.append(f'<line class="axis0" x1="0" x2="{w}" y1="{mid:.1f}" y2="{mid:.1f}"/>'
               f'<text class="pane-t" x="2" y="{y2 + 8}">MACD</text>')
    segs = {"vu": [], "vd": []}
    for i, v in enumerate(hist):
        hgt = abs(v) / hm * (mh / 2 - 2)
        segs["vu" if v >= 0 else "vd"].append(f"M{x(i):.1f} {mid:.1f}v{-hgt if v >= 0 else hgt:.1f}")
    svg += [f'<path class="{k}b" style="stroke-width:{bw:.1f}" d="{"".join(v)}"/>' for k, v in segs.items() if v]
    fmt = "%y/%m/%d" if n > 200 else "%m/%d"
    return (f'<svg viewBox="0 0 {w} {H}" class="chart" role="img" aria-label="최근 {len(d)}일 가격·거래량·RSI·MACD 차트">'
            f'{"".join(svg)}</svg><div class="axis"><span>{d.index[0].strftime(fmt)}</span>'
            f'<span>{d.index[-1].strftime(fmt)}</span></div>')


def intraday_chart(df: pd.DataFrame, cur: str = "KRW") -> str:
    """코인 1일 차트: 15분봉 캔들. 일봉 점수와는 별개로 흐름만 보는 용도."""
    w, h, pad = 320, 170, 18
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    span = (hi - lo) or 1
    step = w / len(df)
    bw = max(step * 0.6, 1.2)

    def y(v: float) -> float:
        return pad + (hi - v) / span * (h - 2 * pad)

    svg = []
    for i, (_, r) in enumerate(df.iterrows()):
        cx = step / 2 + i * step
        c = "cu" if r["Close"] >= r["Open"] else "cd"
        top, bot = y(max(r["Open"], r["Close"])), y(min(r["Open"], r["Close"]))
        svg.append(f'<line class="{c}" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(r["High"]):.1f}" y2="{y(r["Low"]):.1f}"/>'
                   f'<rect class="{c}" x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(bot - top, 0.8):.1f}"/>')
    hi_i, lo_i = int(df["High"].values.argmax()), int(df["Low"].values.argmin())
    for idx_, val, up in ((hi_i, hi, False), (lo_i, lo, True)):
        cx = min(max(step / 2 + idx_ * step, 18), w - 18)
        svg.append(f'<text class="pl {"pb" if up else "ps"}" x="{cx:.1f}" y="{y(val) + (12 if up else -5):.1f}" '
                   f'text-anchor="middle">{"저" if up else "고"} {man(val, cur)}</text>')
    chg = (df["Close"].iloc[-1] / df["Open"].iloc[0] - 1) * 100
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="최근 24시간 15분봉 차트">{"".join(svg)}</svg>'
            f'<div class="axis"><span>{df.index[0]:%m/%d %H:%M}</span><span>24시간 {chg:+.2f}%</span>'
            f'<span>{df.index[-1]:%H:%M}</span></div>'
            f'<p class="muted small">1일 차트는 흐름만 보는 용도예요. 점수는 일봉 기준이에요.</p>')


PERIODS = [("1w", "1주"), ("1m", "1달"), ("3m", "3달"), ("6m", "6달"), ("1y", "1년")]


def period_bars(mid: str) -> dict:
    if mid == "coin":  # 코인은 주말에도 거래
        return {"1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
    return {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 250}


def charts_html(ind: pd.DataFrame, marks: list[dict], mid: str, cur: str, lv: dict,
                intraday: pd.DataFrame | None) -> str:
    out = []
    if mid == "coin":
        inner = intraday_chart(intraday, cur) if intraday is not None and len(intraday) > 5 \
            else '<p class="muted small">1일 차트를 받지 못했어요.</p>'
        out.append(f'<div class="ch" data-p="1d">{inner}</div>')
    for key, n in period_bars(mid).items():
        style = "candle" if key in ("1w", "1m") else "line"
        out.append(f'<div class="ch" data-p="{key}">{chart(ind, marks, n, cur, style, lv, label_all=n <= 100)}</div>')
    return "".join(out)


def recent_marks_html(marks: list[dict], now_price: float, k: int = 6, cur: str = "KRW") -> str:
    items = []
    for m in reversed(marks[-k:]):
        t = ("🔥 강한 매수" if m["strong"] else "▲ 매수") if m["kind"] == "buy" else ("🐵 강한 매도" if m["strong"] else "▼ 매도")
        diff = (now_price / m["price"] - 1) * 100
        items.append(f'<li><b class="{m["kind"]}">{m["day"]} {t}</b> <b>{money(m["price"], cur)}</b> '
                     f'<span class="{"up" if diff >= 0 else "down"}">(지금 {diff:+.1f}%)</span>'
                     f'<br><span class="muted">{html.escape(m["label"])}</span></li>')
    return f'<ul class="hist">{"".join(items)}</ul>' if items else '<p class="muted small">최근 1년 신호 없음</p>'


# ---------------------------------------------------------------- 대시보드: 카드

def score_block(sc: dict, title: str, kind: str, open_: bool = False) -> str:
    e = html.escape
    if sc["value"] is None:
        return f'<div class="sb"><b>{title}</b> <span class="muted">데이터 부족</span></div>'
    pct = sc["value"] / sc["scale"] * 100
    good = pct >= 75 if kind == "buy" else pct < 45
    bad = pct < 50 if kind == "buy" else pct >= 65
    cls = "good" if good else "bad" if bad else "mid"
    rows = []
    for it in sc["items"]:
        if it["score"] is None:
            rows.append(f'<tr class="na"><td>{e(it["name"])}</td><td>N/A</td><td>{e(it["reason"])}</td></tr>')
        else:
            w = it["score"] / it["max"] * 100
            rows.append(f'<tr><td>{e(it["name"])}</td><td><b>{it["score"]:g}</b>/{it["max"]}'
                        f'<div class="bar"><i class="{kind}" style="width:{w:.0f}%"></i></div></td>'
                        f'<td>{e(it["reason"])}</td></tr>')
    note = "" if sc["confidence"] == 100 else f" · 신뢰도 {sc['confidence']}% (N/A 항목 제외 환산)"
    if sc.get("note"):
        note += f" · {sc['note']}"
    return (f'<div class="sb {cls}"><div class="sbh"><span>{title}</span><b>{sc["value"]:.0f}<small>/{sc["scale"]}</small></b>'
            f'<em>{e(sc["grade"])}</em></div><div class="gauge"><i class="{kind}" style="width:{pct:.0f}%"></i></div>'
            f'<details{" open" if open_ else ""}><summary>항목별 점수와 이유{note}</summary><table class="items">{"".join(rows)}</table></details></div>')


def plan_block(r: dict, cur: str) -> str:
    p = r["plan"]
    cells = []
    if p.get("buy_zone"):
        lo_, hi_ = p["buy_zone"]
        z2 = f' · 2차 {price_txt(p["buy_zone2"], cur)}' if p.get("buy_zone2") else ""
        cells.append(f'<div><label>분할매수 구간</label><b class="lo">{price_txt(lo_, cur)}~{price_txt(hi_, cur)}</b>'
                     f'<small>{z2}</small></div>')
    cells.append(f'<div><label>손절 고려가</label><b class="hi">{price_txt(p["stop"], cur)}</b>'
                 f'<small>지지선 − 하루 평균 변동폭</small></div>')
    cells.append(f'<div><label>1차·2차 목표</label><b>{price_txt(p["target1"], cur)} / {price_txt(p["target2"], cur)}</b>'
                 f'<small>{p["target_basis"]} 기준</small></div>')
    if r.get("qty"):
        if r.get("target"):
            cells.append(f'<div><label>내 목표가까지</label><b>{(r["target"] / r["close"] - 1) * 100:+.1f}%</b></div>')
        if r.get("stop"):
            cells.append(f'<div><label>내 손절가까지</label><b>{(r["stop"] / r["close"] - 1) * 100:+.1f}%</b></div>')
    return f'<div class="plan">{"".join(cells)}</div>'


def render_cards(rows: list[dict], cur: str, show_pf: bool) -> str:
    e = html.escape

    def badge(r: dict) -> str:
        s, out = r["signal"], []
        if s["buy_level"]:
            out.append(f'<span class="b buy">{"🔥 강한 " if s["buy_level"] == "strong" else ""}매수 타이밍</span>')
        if s["sell_level"]:
            out.append(f'<span class="b sell">{"🐵 강한 " if s["sell_level"] == "strong" else ""}매도 경고</span>')
        return "".join(out) or '<span class="b none">관망</span>'

    def order(r: dict) -> float:
        if r.get("error"):
            return 999
        s = r["signal"]
        return -(2 if s["sell_level"] == "strong" or s["buy_level"] == "strong" else 1 if (s["sell_level"] or s["buy_level"]) else 0) * 100 \
            - (r["timing"]["value"] or 0)

    cards = []
    for r in sorted(rows, key=order):
        if r.get("error"):
            cards.append(f'<div class="card"><h2>{e(r["name"])}</h2><p class="muted">데이터를 받지 못했어요: {e(r["error"])}</p></div>')
            continue
        pf = ""
        if show_pf and r.get("qty"):
            pf = (f'<div class="pf"><span>평단 {money(r["avg"], cur)}</span>'
                  f'<span class="{"up" if r["ret"] >= 0 else "down"}">{r["ret"]:+.1f}% ({money(r["pnl"], cur)})</span></div>')
        news = "".join(f'<li><a href="{e(n["link"])}" target="_blank" rel="noopener">{e(n["title"])}</a></li>'
                       for n in r["news"][:3])
        risk_title = "매도 위험" + ("" if r.get("qty") else " <small>(보유 시 참고)</small>")
        sig = r["signal"]
        cards.append(f"""
<div class="card">
  <div class="top"><h2>{e(r["name"])} <small>{e(r["code"])}</small></h2>{badge(r)}</div>
  <div class="price"><b>{price_txt(r["close"], cur)}</b><span class="{"up" if r["chg"] >= 0 else "down"}">{r["chg"]:+.2f}%</span>
    <span class="hl">고 {price_txt(r["day_hi"], cur)} · 저 {price_txt(r["day_lo"], cur)}</span></div>
  {pf}
  <div class="why {"sb2" if sig["buy_level"] else "ss" if sig["sell_level"] else ""}">{e(summary_line(r))}</div>
  <div class="scores">{score_block(r["timing"], "매수 타이밍", "buy", bool(sig["buy_level"]))}{score_block(r["risk"], risk_title, "sell", bool(sig["sell_level"]))}</div>
  {plan_block(r, cur)}
  {r["spark"]}
  <details><summary>최근 매매 타이밍 기록</summary>{r["hist"]}</details>
  {f'<details><summary>최근 뉴스</summary><ul class="news">{news}</ul></details>' if news else ""}
</div>""")
    return "".join(cards)


def render_dashboard(markets: list[dict], now: dt.datetime, show_pf: bool) -> str:
    e = html.escape
    tabs, sections = [], []
    for i, m in enumerate(markets):
        rows, fg, idx = m["rows"], m["fg"], m["idx"]
        n_sig = sum(1 for r in rows if not r.get("error") and (r["signal"]["buy_level"] or r["signal"]["sell_level"]))
        fg_txt = f'{fg["score"]:.0f} <small>{e(fg["rating"])}</small>' if fg else "확인 불가"
        fg_cls = "" if not fg else ("lo" if fg["score"] <= 25 else "hi" if fg["score"] >= 75 else "")
        fg_label = (fg or {}).get("label") or m.get("fg_label", "공포탐욕")
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
                    f'<div><label>평가금액</label><b>{money(val, m["cur"])}</b></div>'
                    f'<div><label>총 손익</label><b class="{"up" if val >= cost else "down"}">{(val / cost - 1) * 100:+.1f}%</b></div>'
                )
        badge = f'<span class="cnt">{n_sig}</span>' if n_sig else ""
        tabs.append(f'<button class="tab{" on" if i == 0 else ""}" data-tab="{m["id"]}">{e(m["name"])}{badge}</button>')
        sections.append(f"""
<section id="{m["id"]}" class="pane" data-p="3m"{"" if i == 0 else " hidden"}>
<div class="periods">{"".join(f'<button class="per{" on" if k == "3m" else ""}" data-p="{k}">{lab}</button>' for k, lab in ([("1d", "1일")] if m["id"] == "coin" else []) + PERIODS)}</div>
<div class="sum">
  <div><label>{e(fg_label)}</label><b class="{fg_cls}">{fg_txt}</b></div>
  {idx_html}{total_html}
</div>
<div class="muted small">신호 {n_sig}개 · {e(m.get("note", ""))}</div>
{render_cards(rows, m["cur"], show_pf)}
</section>""")
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
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:6px;margin-top:8px}}.grid b{{font-size:17px}}
.chart{{width:100%;height:auto;margin-top:6px;overflow:visible}}.chart polyline{{fill:none;stroke-width:1.4}}
.chart .px{{stroke:var(--fg);stroke-width:1.8}}.chart .bb{{stroke:var(--mut);stroke-dasharray:3 3;opacity:.6}}
.chart .ma5{{stroke:#ec4899;opacity:.8}}.chart .ma20{{stroke:#f59e0b}}.chart .ma60{{stroke:#8b5cf6}}
.mk-buy{{fill:var(--buy)}}.mk-sell{{fill:var(--sell)}}.mk-gc{{fill:#facc15;stroke:#a16207;stroke-width:1}}.mk-dc{{fill:#64748b;stroke:#0f172a;stroke-width:1}}
.chart .emo{{font-size:13px}}.chart .pl{{font-size:9px;font-weight:700;paint-order:stroke;stroke:var(--card);stroke-width:3px}}
.chart .pb{{fill:var(--buy)}}.chart .ps{{fill:var(--sell)}}.hist li{{margin-bottom:4px}}
.axis{{display:flex;justify-content:space-between;font-size:11px;color:var(--mut)}}
.legend{{font-size:12px;color:var(--mut);display:flex;flex-wrap:wrap;gap:4px 10px;margin:6px 0}}.legend i{{font-style:normal;font-weight:700}}
.hist{{margin:6px 0 0;padding-left:16px;font-size:13px}}.hist .buy{{color:var(--buy)}}.hist .sell{{color:var(--sell)}}.hist .gc{{color:#ca8a04}}.hist .dc{{color:#64748b}}.small{{font-size:13px}}
.mas{{font-size:13px;margin-top:8px}}.mas span{{margin-right:6px;font-weight:600}}
.reasons{{margin:4px 0 0;padding-left:18px;font-size:14px}}.reasons li{{margin-bottom:6px}}.reasons li span{{display:block;color:var(--fg);font-size:13px;opacity:.85}}
.rtitle{{font-size:13px;font-weight:700;margin-top:10px}}
.why{{font-size:14px;background:var(--bg);border-radius:10px;padding:8px 10px;margin:8px 0}}.why.sb2{{border-left:4px solid var(--buy)}}.why.ss{{border-left:4px solid var(--sell)}}li.buy{{color:var(--buy)}}li.sell{{color:var(--sell)}}
.pf{{display:flex;justify-content:space-between;font-size:14px}}details{{margin-top:8px;font-size:14px}}.news a{{color:inherit}}
footer{{font-size:12px;color:var(--mut);margin:20px 0;line-height:1.6}}
.chart .ma120{{stroke:#0ea5e9}}.chart line.sup,.chart line.res{{stroke-dasharray:4 3;stroke-width:1}}.chart line.sup{{stroke:var(--buy)}}.chart line.res{{stroke:var(--sell)}}
.chart .lvl{{font-size:8px;font-weight:700;paint-order:stroke;stroke:var(--card);stroke-width:3px}}.chart .supt{{fill:var(--buy)}}.chart .rest{{fill:var(--sell)}}
.chart .pane-t{{font-size:8px;fill:var(--mut)}}.chart .vub{{stroke:var(--up);opacity:.55;fill:none}}.chart .vdb{{stroke:var(--down);opacity:.55;fill:none}}
.chart .zone{{fill:var(--mut);opacity:.08}}.chart .rsi{{stroke:#8b5cf6;stroke-width:1.2}}.chart .axis0{{stroke:var(--line)}}
.scores{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0}}@media (max-width:520px){{.scores{{grid-template-columns:1fr}}}}
.sb{{border:1px solid var(--line);border-radius:12px;padding:10px}}.sb.good{{border-color:var(--buy)}}.sb.bad{{border-color:var(--sell)}}
.sbh{{display:flex;align-items:baseline;gap:6px}}.sbh span{{font-size:13px;font-weight:700}}.sbh b{{font-size:22px;margin-left:auto}}.sbh em{{font-style:normal;font-size:12px;color:var(--mut)}}
.gauge,.bar{{height:6px;background:var(--line);border-radius:9px;overflow:hidden;margin-top:4px}}.bar{{height:4px;width:60px}}
.gauge i,.bar i{{display:block;height:100%}}i.buy{{background:var(--buy)}}i.sell{{background:var(--sell)}}
.items{{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}}.items td{{border-top:1px solid var(--line);padding:5px 4px;vertical-align:top}}
.items td:first-child{{white-space:nowrap;font-weight:600}}.items td:nth-child(2){{white-space:nowrap}}.items tr.na{{color:var(--mut)}}
.plan{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:6px;margin:6px 0}}.plan div{{background:var(--bg);border-radius:10px;padding:7px 9px}}
.plan b{{display:block;font-size:14px}}.plan small{{font-size:11px}}.hist .buy{{color:var(--buy)}}.hist .sell{{color:var(--sell)}}
.tabs{{position:sticky;top:0;z-index:5;display:flex;gap:6px;background:var(--bg);padding:10px 0;margin-top:6px}}
.tab{{flex:1;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;padding:9px 0;font:inherit;font-weight:700;cursor:pointer}}
.pane{{scroll-margin-top:64px}}
.periods{{display:flex;gap:4px;margin:2px 0 8px;overflow-x:auto}}.per{{flex:1;min-width:44px;border:1px solid var(--line);background:var(--card);color:var(--mut);border-radius:99px;padding:5px 0;font:inherit;font-size:13px;cursor:pointer}}
.per.on{{color:var(--fg);border-color:var(--fg);font-weight:700}}
.ch{{display:none}}.pane[data-p="1d"] .ch[data-p="1d"]{{display:block}}.pane[data-p="1w"] .ch[data-p="1w"]{{display:block}}.pane[data-p="1m"] .ch[data-p="1m"]{{display:block}}.pane[data-p="3m"] .ch[data-p="3m"]{{display:block}}.pane[data-p="6m"] .ch[data-p="6m"]{{display:block}}.pane[data-p="1y"] .ch[data-p="1y"]{{display:block}}
.chart .cu{{fill:var(--up);stroke:var(--up)}}.chart .cd{{fill:var(--down);stroke:var(--down)}}.chart line.cu,.chart line.cd{{stroke-width:1}}
.hl{{font-size:12px;color:var(--mut);margin-left:auto}}.tab.on{{background:var(--fg);color:var(--bg);border-color:var(--fg)}}.cnt{{display:inline-block;margin-left:5px;background:var(--sell);color:#fff;border-radius:99px;font-size:11px;padding:0 6px}}
</style></head><body><main>
<h1>내 주식 타이밍 보드</h1>
<div class="muted">{now:%Y-%m-%d %H:%M} 기준 · 10분마다 새로고침</div>
<nav class="tabs">{"".join(tabs)}</nav>
<div class="legend"><span><i style="color:var(--buy)">▲</i> 매수 타이밍(75점↑)</span><span>🔥 강한 매수(85점↑)</span><span><i style="color:var(--sell)">▼</i> 매도 경고(65점↑)</span><span>🐵 강한 매도(80점↑)</span>
<span><i style="color:#f59e0b">―</i> 20일선</span><span><i style="color:#8b5cf6">―</i> 60일선</span><span><i style="color:#0ea5e9">―</i> 120일선</span><span><i style="color:var(--buy)">┄</i> 지지선</span><span><i style="color:var(--sell)">┄</i> 저항선</span><span>┄ 볼린저</span></div>
{"".join(sections)}
<footer><b>점수 기준</b> · 매수 타이밍 100점 = 중장기 추세 25 + 모멘텀(RSI·MACD) 20 + 거래량 15 + 수급 15 + 지지·저항 15 + 시장 심리 10
(85↑ 매우 좋음 · 75↑ 좋음 · 65↑ 관심 · 50↑ 관망 · 50 미만 나쁨).
매도 위험 100점 = 추세 약화 20 + 과열·둔화 15 + 거래량·수급 15 + 시장 위험 10 + 밸류에이션 20 + 펀더멘털 20(재무 데이터 연결 전이라 보류, 나머지로 환산).
매도 경고 알림은 보유 종목에만 보내요. 차트의 ▲▼🔥🐵는 과거 각 날짜에 그날까지의 차트 데이터로 같은 점수를 다시 계산한 결과예요(과거 수급·공포탐욕은 제외).
차트 기반 참고 분석이며 투자 판단은 본인 몫입니다.</footer>
</main>
<script>
(function () {{
  var tabs = document.querySelectorAll(".tab"), panes = document.querySelectorAll(".pane");
  function show(id) {{
    var found = false;
    panes.forEach(function (p) {{ var on = p.id === id; p.hidden = !on; found = found || on; }});
    if (!found) return;
    tabs.forEach(function (t) {{ t.classList.toggle("on", t.dataset.tab === id); }});
    try {{ history.replaceState(null, "", "#" + id); }} catch (e) {{}}
  }}
  tabs.forEach(function (t) {{ t.addEventListener("click", function () {{ show(t.dataset.tab); }}); }});
  function setPeriod(pane, p) {{
    if (!pane.querySelector('.per[data-p="' + p + '"]')) return;
    pane.dataset.p = p;
    pane.querySelectorAll(".per").forEach(function (b) {{ b.classList.toggle("on", b.dataset.p === p); }});
    try {{ localStorage.setItem("period-" + pane.id, p); }} catch (e) {{}}
  }}
  panes.forEach(function (pane) {{
    pane.querySelectorAll(".per").forEach(function (b) {{
      b.addEventListener("click", function () {{ setPeriod(pane, b.dataset.p); }});
    }});
    try {{ var saved = localStorage.getItem("period-" + pane.id); if (saved) setPeriod(pane, saved); }} catch (e) {{}}
  }});
  if (location.hash) show(location.hash.slice(1));
}})();
</script>
</body></html>"""




# ---------------------------------------------------------------- 실행

def load_portfolio() -> dict:
    """PORTFOLIO_JSON 예: {"018260":{"qty":18,"avg":181864,"target":260000,"stop":200000}}"""
    raw = os.getenv("PORTFOLIO_JSON", "").strip()
    if not raw and (ROOT / "portfolio.json").exists():
        raw = (ROOT / "portfolio.json").read_text(encoding="utf-8")
    if not raw:
        return {}
    data = json.loads(raw)
    return {(str(k).zfill(6) if str(k).isdigit() else str(k).upper()): v for k, v in data.items()}


def analyze(stock: dict, df: pd.DataFrame, pf: dict, fg: dict | None, news: list[dict],
            inv: list[dict] | None = None, now: dt.datetime | None = None, market: dict | None = None,
            intraday: pd.DataFrame | None = None, idx_chg: float | None = None) -> dict:
    market = market or {"id": "kr", "name": "국장", "cur": "KRW"}
    mid, cur = market["id"], market["cur"]
    ind = scoring.add_indicators(compute_indicators(df, CONFIG["indicators"]))
    last, prev = ind.iloc[-1], ind.iloc[-2]
    now = now or dt.datetime.now(KST)
    scale = session_elapsed(now, mid) if ind.index[-1].date() == market_today(now, mid) else 1.0
    lv = scoring.levels(ind)
    timing = scoring.timing_score(ind, lv, fg, inv, mid, scale)
    risk = scoring.sell_risk(ind, lv, fg, inv, idx_chg, mid, scale)
    plan = scoring.price_plan(ind, lv)
    marks = scoring.history(ind, max(period_bars(mid).values()), mid)
    h = pf.get(stock["code"])
    row = {
        **stock, "mid": mid, "market": market["name"], "cur": cur,
        "asof": ind.index[-1].strftime("%Y-%m-%d"),
        "close": float(last["Close"]),
        "chg": float((last["Close"] / prev["Close"] - 1) * 100),
        "day_hi": float(last["High"]), "day_lo": float(last["Low"]),
        "timing": timing, "risk": risk, "plan": plan,
        "signal": signal_from_scores(timing, risk, held=bool(h)),
        "news": news,
        "spark": charts_html(ind, marks, mid, cur, lv, intraday),
        "hist": recent_marks_html(marks, float(last["Close"]), cur=cur),
    }
    if h:
        qty, avg = float(h["qty"]), float(h["avg"])
        row.update(qty=qty, avg=avg, ret=(row["close"] / avg - 1) * 100, pnl=(row["close"] - avg) * qty,
                   target=h.get("target"), stop=h.get("stop"))
    return row


def load_markets() -> list[dict]:
    """config.json의 markets (예전 형식의 stocks만 있으면 국장으로 취급)."""
    if "markets" in CONFIG:
        return CONFIG["markets"]
    return [{"id": "kr", "name": "국장", "cur": "KRW", "source": "fdr", "stocks": CONFIG["stocks"]}]


def market_context(m: dict) -> tuple[dict | None, dict]:
    """시장별 공포탐욕지수와 지수 등락률."""
    if m["id"] == "coin":
        fg = fetch_crypto_fear_greed()
        if fg:
            fg["label"] = "코인 공포탐욕"
        return fg, {}
    fg = fetch_fear_greed()
    if fg:
        fg["label"] = "공포탐욕(CNN)"
    if m["id"] == "us":
        return fg, {"S&P500": fetch_index_change("US500"), "나스닥": fetch_index_change("IXIC")}
    return fg, {"코스피": fetch_index_change("KS11"), "코스닥": fetch_index_change("KQ11")}


def main() -> int:
    now = dt.datetime.now(KST)
    pf = load_portfolio()
    show_pf = os.getenv("SHOW_PORTFOLIO", "false").lower() == "true"

    markets = []
    for m in load_markets():
        m = {"cur": "KRW", "source": "fdr", **m}
        fg, idx = market_context(m)
        idx_chg = next((v for v in idx.values() if v is not None), None)
        rows = []
        for s in m["stocks"]:
            try:
                df = fetch_ohlcv(s["code"], m["source"])
                if len(df) < 130:
                    raise ValueError(f"일봉 {len(df)}개뿐")
                q = fetch_quote(s["code"]) if m["id"] == "kr" else None
                inv = fetch_investor(s["code"]) if m["id"] == "kr" else None
                news = fetch_news(s.get("news") or s["name"], suffix=m.get("news_suffix", "주가"))
                intra = fetch_upbit_intraday(s["code"]) if m["id"] == "coin" else None
                row = analyze(s, apply_quote(df, q), pf, fg, news, inv, now, m, intra, idx_chg)
                if q:  # 등락률은 증권사 앱과 같은 공식 값 사용
                    row["chg"] = q["chg"]
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {m['name']} {s['name']} 실패: {e}")
                rows.append({**s, "mid": m["id"], "error": str(e)[:80]})
        markets.append({**m, "rows": rows, "fg": fg, "idx": idx})

    # 알림: 같은 날 같은 신호는 1번만, 일반→강한(🔥/🐵)으로 올라가면 다시 알림
    state = load_state()
    sent_any = 0
    for m in markets:
        for r in m["rows"]:
            if r.get("error"):
                continue
            for side in ("buy", "sell"):
                lvl = r["signal"][f"{side}_level"]
                if not lvl:
                    continue
                key = f"{r['asof']}:{r['code']}:{side}"
                if state.get(key) == lvl or (state.get(key) == "strong" and lvl == "normal"):
                    continue
                if send_telegram(build_alert(r, side, m["fg"], m["idx"])):
                    sent_any += 1
                state[key] = lvl
            # 보유 종목: 내가 정한 목표가·손절가 도달 (단순 가격 알림)
            for kind in ("target", "stop"):
                lvl_price = r.get(kind)
                if not lvl_price:
                    continue
                hit = r["close"] >= lvl_price if kind == "target" else r["close"] <= lvl_price
                key = f"{r['asof']}:{r['code']}:{kind}"
                if hit and key not in state:
                    if send_telegram(build_price_alert(r, kind)):
                        sent_any += 1
                    state[key] = "hit"
    if state:  # 최근 7일치만 보관
        newest = dt.date.fromisoformat(max(k[:10] for k in state))
        cutoff = (newest - dt.timedelta(days=7)).isoformat()
        state = {k: v for k, v in state.items() if k[:10] >= cutoff}
    save_state(state)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(render_dashboard(markets, now, show_pf), encoding="utf-8")
    total = sum(len(m["rows"]) for m in markets)
    print(f"완료: {len(markets)}개 시장 {total}종목, 새 알림 {sent_any}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
