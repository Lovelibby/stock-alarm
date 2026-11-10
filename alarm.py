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

    start = (dt.datetime.now(KST) - dt.timedelta(days=400)).strftime("%Y-%m-%d")
    df = fdr.DataReader(code, start)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    return df.dropna().astype(float)


def fetch_upbit(market: str, count: int = 200) -> pd.DataFrame:
    """업비트 원화 마켓 일봉 (하루 = 매일 오전 9시 시작). 현재 진행 중인 봉 포함."""
    import requests

    r = requests.get(
        "https://api.upbit.com/v1/candles/days",
        params={"market": market, "count": count},
        headers={"accept": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df.index = pd.to_datetime(df["candle_date_time_kst"].str[:10])
    df = df.rename(columns={"opening_price": "Open", "high_price": "High", "low_price": "Low",
                            "trade_price": "Close", "candle_acc_trade_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index().astype(float)


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


# ---------------------------------------------------------------- 신호 판정

def crosses(last: pd.Series, prev: pd.Series) -> list[tuple[str, str]]:
    """(방향, 이름) 목록. 단기 5/20, 중기 20/60 골든·데드크로스."""
    out = []
    for a, b, tag in ((5, 20, "단기"), (20, 60, "중기")):
        pa, pb, la, lb = prev[f"MA{a}"], prev[f"MA{b}"], last[f"MA{a}"], last[f"MA{b}"]
        if np.isnan([pa, pb, la, lb]).any():
            continue
        if pa <= pb and la > lb:
            out.append(("buy", f"{a}일선↗{b}일선 골든크로스({tag})"))
        if pa >= pb and la < lb:
            out.append(("sell", f"{a}일선↘{b}일선 데드크로스({tag})"))
    return out


def tech_reasons(last: pd.Series, prev: pd.Series, vol_scale: float = 1.0) -> tuple[list[str], list[str]]:
    """종목 자체 기술적 지표로 매수·매도 근거 목록을 만듦."""
    t = CONFIG["thresholds"]
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

    surge = t.get("volume_surge", 2.0)
    if last["VOL20"] and not np.isnan(last["VOL20"]) and last["Volume"] > 0:
        ratio = last["Volume"] / vol_scale / last["VOL20"]
        if ratio >= surge:
            if close >= prev["Close"]:
                buy.append(f"거래량 급증 {ratio:.1f}배 + 상승")
            else:
                sell.append(f"거래량 급증 {ratio:.1f}배 + 하락")

    cx = crosses(last, prev)
    buy += [n for d, n in cx if d == "buy"]
    sell += [n for d, n in cx if d == "sell"]
    if not any(d == "buy" and "단기" in n for d, n in cx) and prev["Close"] < prev["MA20"] and close >= last["MA20"]:
        buy.append("20일선 회복")
    if not any(d == "sell" and "단기" in n for d, n in cx) and prev["Close"] > prev["MA20"] and close < last["MA20"]:
        sell.append("20일선 이탈")
    return buy, sell


def evaluate(ind: pd.DataFrame, fg: dict | None, nscore: int, inv: list[dict] | None = None,
             vol_scale: float = 1.0) -> dict:
    t = CONFIG["thresholds"]
    last, prev = ind.iloc[-1], ind.iloc[-2]
    buy, sell = tech_reasons(last, prev, vol_scale)

    k = t.get("investor_days", 3)
    if inv and len(inv) >= k:
        net = [d["foreign"] + d["inst"] for d in inv[:k]]
        if all(x > 0 for x in net):
            buy.append(f"외국인·기관 {k}일 연속 순매수")
        elif all(x < 0 for x in net):
            sell.append(f"외국인·기관 {k}일 연속 순매도")

    # 여기까지가 종목 자체 기술적 지표. 아래(시장 심리·뉴스)는 보조 가점이라
    # 기술적 지표가 하나도 없으면 신호를 내지 않음 (시장 공포만으로 전 종목 매수 신호가 뜨는 것 방지)
    tech_buy, tech_sell = len(buy), len(sell)

    if fg:
        if fg["score"] <= t["fear_extreme"]:
            buy.append(f"{fg.get('label', '공포탐욕')} {fg['score']:.0f} 극단적 공포")
        if fg["score"] >= t["greed_extreme"]:
            sell.append(f"{fg.get('label', '공포탐욕')} {fg['score']:.0f} 극단적 탐욕")

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


# ---------------------------------------------------------------- 쉬운 설명

EXPLAIN = [
    ("RSI", "과매도", "최근에 너무 빨리 많이 떨어진 상태예요. 이런 구간에서는 잠깐이라도 반등이 나오는 경우가 많아요."),
    ("RSI", "과매수", "짧은 기간에 너무 많이 올라 과열된 상태예요. 쉬어가거나 조금 되돌아오기 쉬워요."),
    ("CCI", "과매도", "주가가 평소 움직이던 범위보다 한참 아래로 내려왔어요. 원래 범위로 돌아오려는 힘이 생기기 쉬워요."),
    ("CCI", "과매수", "주가가 평소 움직이던 범위보다 한참 위로 올라왔어요. 힘이 빠지면 내려오기 쉬워요."),
    ("볼린저 하단", "", "주가가 평소 오르내리는 '통로'의 바닥에 닿았어요. 보통은 통로 안쪽으로 다시 올라오려 해요."),
    ("볼린저 상단", "", "주가가 평소 오르내리는 '통로'의 천장에 닿았어요. 보통은 통로 안쪽으로 다시 내려오려 해요."),
    ("골든크로스(단기)", "", "최근 5일 평균값이 20일 평균값 위로 올라섰어요. 몇 주 단위 흐름이 '하락→상승'으로 바뀌는 신호예요."),
    ("데드크로스(단기)", "", "최근 5일 평균값이 20일 평균값 아래로 내려갔어요. 몇 주 단위 흐름이 '상승→하락'으로 바뀌는 신호예요."),
    ("골든크로스(중기)", "", "20일 평균값이 60일 평균값 위로 올라섰어요. 몇 달 단위 흐름이 좋아지는 신호라 단기보다 무게가 커요."),
    ("데드크로스(중기)", "", "20일 평균값이 60일 평균값 아래로 내려갔어요. 몇 달 단위 흐름이 나빠지는 신호라 단기보다 무게가 커요."),
    ("20일선 회복", "", "한 달 평균 가격 아래 있던 주가가 다시 위로 올라왔어요. 하락을 멈추고 힘을 되찾는 모습이에요."),
    ("20일선 이탈", "", "한 달 평균 가격 위에 있던 주가가 아래로 내려갔어요. 오르던 힘이 약해졌다는 뜻이에요."),
    ("거래량 급증", "상승", "평소보다 훨씬 많은 사람이 사고팔면서 올랐어요. 거래가 실린 상승은 진짜 힘이 있는 경우가 많아요."),
    ("거래량 급증", "하락", "평소보다 훨씬 많은 물량이 쏟아지며 떨어졌어요. 누군가 크게 팔고 나가는 중일 수 있어요."),
    ("순매수", "", "외국인·기관 같은 '큰손'이 며칠째 사 모으고 있어요. 큰손이 사는 종목은 오르는 힘이 오래가는 편이에요."),
    ("순매도", "", "외국인·기관 같은 '큰손'이 며칠째 팔고 있어요. 큰손이 파는 동안은 오르기 어려운 편이에요."),
    ("극단적 공포", "", "시장 전체가 겁에 질려 있어요. 역사적으로 이럴 때 싸게 살 기회가 많았어요 (보조 지표)."),
    ("극단적 탐욕", "", "시장 전체가 너무 들떠 있어요. 이럴 때는 조심하는 게 좋아요 (보조 지표)."),
    ("호재 뉴스", "", "최근 2일 기사 중 좋은 소식이 더 많아요 (보조 지표)."),
    ("악재 뉴스", "", "최근 2일 기사 중 나쁜 소식이 더 많아요 (보조 지표)."),
]


def explain(reason: str) -> str:
    for a, b, text in EXPLAIN:
        if a in reason and b in reason:
            return text
    return ""


def summary_line(sig: dict) -> str:
    """카드 맨 위에 보여줄 한 줄 결론."""
    nb, ns = len(sig["buy"]), len(sig["sell"])
    need = CONFIG["signal"]["min_score"]
    if sig["buy_level"] and sig["sell_level"]:
        return f"사라는 근거 {nb}개와 팔라는 근거 {ns}개가 섞여 있어요. 방향이 엇갈리니 서두르지 않는 게 좋아요."
    if sig["buy_level"] == "strong":
        return f"🔥 사라는 근거가 {nb}개나 겹쳤어요. 이 기준에서 가장 강한 매수 타이밍이에요. 한 번에 다 사기보다 나눠서 사는 걸 고려해 보세요."
    if sig["buy_level"]:
        return f"사라는 근거가 {nb}개 겹쳐 매수 타이밍이에요. 다만 더 떨어질 수도 있어서 나눠서 사는 게 안전해요."
    if sig["sell_level"] == "strong":
        return f"🐵 팔라는 근거가 {ns}개나 겹쳤어요. 이 기준에서 가장 강한 매도 타이밍이에요. 수익 중이면 일부라도 챙기는 걸 고려해 보세요."
    if sig["sell_level"]:
        return f"팔라는 근거가 {ns}개 겹쳐 매도 타이밍이에요. 전부 팔기보다 일부 매도로 대응하는 방법도 있어요."
    if nb or ns:
        return (f"매수 근거 {nb}개 · 매도 근거 {ns}개로, 신호 기준({need}개)에 아직 못 미쳐요. "
                "지금은 지켜보는 구간이에요.")
    return "특별히 걸리는 지표가 없어요. 평소 흐름대로 움직이는 중이라 지켜보는 구간이에요."


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
    return f"{x:,.0f}원" if x >= 100 else f"{x:,.2f}원"


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
    sig = row["signal"]
    esc = lambda x: html.escape(x, quote=False)  # noqa: E731
    reasons = sig[side]
    strong = sig[f"{side}_level"] == "strong"
    label = "매수" if side == "buy" else "매도"
    icon = "🟢" if side == "buy" else "🔴"
    big = (" 🔥" if side == "buy" else " 🐵") if strong else ""
    cur = row.get("cur", "KRW")
    tag = f"[{row['market']}] " if row.get("market") else ""
    lines = [f"{icon}{big} <b>{esc(tag)}{'강한 ' if strong else ''}{label} 신호 — {esc(row['name'])}</b>"]
    lines.append(f"현재가 {money(row['close'], cur)} ({row['chg']:+.2f}%)")
    if row.get("qty"):
        lines.append(
            f"평단 {money(row['avg'], cur)} · 수익률 {row['ret']:+.1f}% · 평가손익 {money(row['pnl'], cur)}"
        )
    lines.append("근거: " + ", ".join(esc(r) for r in reasons))
    whys = [explain(r) for r in reasons if explain(r)]
    if whys:
        lines.append("💡 쉽게 말하면")
        lines += [f"• {esc(w)}" for w in whys[:3]]
    lines.append(esc(summary_line(sig)))
    lines.append(
        f"RSI {row['rsi']:.1f} · CCI {row['cci']:.0f} · 볼린저 위치 {row['bb_pos']:.0f}%"
    )
    if row["news"]:
        lines.append("뉴스: " + esc(row["news"][0]["title"][:60]))
    mk = []
    if fg:
        mk.append(f"{fg.get('label', '공포탐욕')} {fg['score']:.0f}")
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

def history_marks(ind: pd.DataFrame, n: int = 60) -> list[dict]:
    """최근 n일 동안 매일 같은 기준(기술적 지표 2개 이상)으로 판정한 과거 신호와 크로스."""
    min_score = CONFIG["signal"]["min_score"]
    marks = []
    was_buy = was_sell = False
    start = max(1, len(ind) - n)
    for i in range(start, len(ind)):
        last, prev = ind.iloc[i], ind.iloc[i - 1]
        if np.isnan(last[["RSI", "CCI", "BB_UP", "MA20"]].astype(float)).any():
            continue
        b, s = tech_reasons(last, prev)
        day = ind.index[i].strftime("%m/%d")
        for d, name in crosses(last, prev):
            marks.append({"i": i - start, "kind": "gc" if d == "buy" else "dc", "label": name, "day": day,
                          "price": float(last["Close"]), "strong": False})
        # 신호가 며칠 연속되면 차트가 화살표로 덮이므로, 신호가 '처음 켜진 날'만 표시
        is_buy, is_sell = len(b) >= min_score, len(s) >= min_score
        strong_score = CONFIG["signal"]["strong_score"]
        px = float(last["Close"])
        if is_buy and not was_buy:
            marks.append({"i": i - start, "kind": "buy", "label": ", ".join(b), "day": day,
                          "price": px, "strong": len(b) >= strong_score})
        if is_sell and not was_sell:
            marks.append({"i": i - start, "kind": "sell", "label": ", ".join(s), "day": day,
                          "price": px, "strong": len(s) >= strong_score})
        was_buy, was_sell = is_buy, is_sell
    return marks


def chart(ind: pd.DataFrame, marks: list[dict], n: int = 60, cur: str = "KRW") -> str:
    d = ind.tail(n)
    cols = ["Close", "BB_UP", "BB_LO", "MA5", "MA20", "MA60"]
    vals = d[cols].stack().dropna()
    lo, hi = float(vals.min()), float(vals.max())
    w, h, pad = 320, 170, 26
    span = (hi - lo) or 1
    step = w / max(len(d) - 1, 1)

    def x(i: int) -> float:
        return i * step

    def y(v: float) -> float:
        return pad + (hi - v) / span * (h - 2 * pad)

    def line(col: str, cls: str) -> str:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(d[col]) if not np.isnan(v))
        return f'<polyline class="{cls}" points="{pts}"/>' if pts else ""

    svg = [line("BB_UP", "bb"), line("BB_LO", "bb"), line("MA60", "ma60"), line("MA20", "ma20"),
           line("MA5", "ma5"), line("Close", "px")]
    for m in marks:
        i = m["i"]
        if i >= len(d):
            continue
        px = float(d["Close"].iloc[i])
        cx = x(i)
        title = "<title>" + html.escape(m["day"] + " " + money(m["price"], cur) + " · " + m["label"]) + "</title>"
        tx = min(max(cx, 14), w - 14)
        if m["kind"] in ("buy", "sell"):
            up = m["kind"] == "buy"  # 매수는 가격 아래, 매도는 가격 위에 표시
            yy = min(y(px) + 12, h - 14) if up else max(y(px) - 12, 14)
            if m.get("strong"):
                svg.append(f'<text class="emo" x="{cx:.1f}" y="{yy + 5:.1f}" text-anchor="middle">'
                           f'{"🔥" if up else "🐵"}{title}</text>')
            elif up:
                svg.append(f'<path class="mk-buy" d="M{cx:.1f},{yy - 6:.1f} l-5,8 h10 z">{title}</path>')
            else:
                svg.append(f'<path class="mk-sell" d="M{cx:.1f},{yy + 6:.1f} l-5,-8 h10 z">{title}</path>')
            ly = yy + 16 if up else yy - 10
            svg.append(f'<text class="pl {"pb" if up else "ps"}" x="{tx:.1f}" y="{ly:.1f}" '
                       f'text-anchor="middle">{man(px, cur)}</text>')
        else:
            ma = float(d["MA20"].iloc[i])
            cls = "mk-gc" if m["kind"] == "gc" else "mk-dc"
            svg.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{y(ma):.1f}" r="4.5">{title}</circle>')
    first, last_day = d.index[0].strftime("%m/%d"), d.index[-1].strftime("%m/%d")
    return (
        f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
        f'aria-label="최근 {n}일 종가·이동평균선·볼린저밴드와 매수·매도 표시">{"".join(svg)}</svg>'
        f'<div class="axis"><span>{first}</span><span>{last_day}</span></div>'
    )


def recent_marks_html(marks: list[dict], now_price: float, k: int = 6, cur: str = "KRW") -> str:
    names = {"buy": ("▲ 매수", "buy"), "sell": ("▼ 매도", "sell"), "gc": ("● 골든크로스", "gc"), "dc": ("● 데드크로스", "dc")}
    items = []
    for m in reversed(marks[-k:]):
        t, cls = names[m["kind"]]
        if m.get("strong"):
            t = ("🔥 강한 매수" if m["kind"] == "buy" else "🐵 강한 매도")
        diff = (now_price / m["price"] - 1) * 100
        items.append(
            f'<li><b class="{cls}">{m["day"]} {t}</b> <b>{money(m["price"], cur)}</b> '
            f'<span class="{"up" if diff >= 0 else "down"}">(지금 {diff:+.1f}%)</span>'
            f'<br><span class="muted">{html.escape(m["label"])}</span></li>'
        )
    return f'<ul class="hist">{"".join(items)}</ul>' if items else '<p class="muted small">최근 60일 신호 없음</p>'


def render_cards(rows: list[dict], cur: str, show_pf: bool) -> str:
    e = html.escape

    def badge(r: dict) -> str:
        s = r["signal"]
        out = []
        if s["buy_level"]:
            out.append(f'<span class="b buy">{"🔥 강한 " if s["buy_level"] == "strong" else ""}매수 신호</span>')
        if s["sell_level"]:
            out.append(f'<span class="b sell">{"🐵 강한 " if s["sell_level"] == "strong" else ""}매도 신호</span>')
        return "".join(out) or '<span class="b none">관망</span>'

    def rsi_cls(v: float) -> str:
        return "lo" if v <= 30 else "hi" if v >= 70 else ""

    def cci_cls(v: float) -> str:
        return "lo" if v <= -100 else "hi" if v >= 100 else ""

    cards = []
    for r in sorted(rows, key=lambda r: 1 if r.get("error") else -(len(r["signal"]["buy"]) + len(r["signal"]["sell"]))):
        if r.get("error"):
            cards.append(f'<div class="card"><h2>{e(r["name"])}</h2><p class="muted">데이터를 받지 못했어요: {e(r["error"])}</p></div>')
            continue

        def li(x: str, cls: str) -> str:
            why = explain(x)
            return f'<li class="{cls}"><b>{e(x)}</b>{f"<span>{e(why)}</span>" if why else ""}</li>'

        reasons = [li(x, "buy") for x in r["signal"]["buy"]] + [li(x, "sell") for x in r["signal"]["sell"]]
        inv = r["inv"][:3]
        if r.get("mid") != "kr":
            inv_html = ""
        elif inv:
            inv_txt = " / ".join(f'{d["foreign"] + d["inst"]:+,.0f}' for d in inv)
            inv_html = f'<div class="mas">외국인+기관 순매수(최근 3일, 주): <b>{e(inv_txt)}</b></div>'
        else:
            inv_html = '<div class="mas muted">외국인·기관 수급: 확인 불가</div>'
        vr = r["vol_ratio"]
        vol_html = "" if np.isnan(vr) else f'<div><label>거래량(평소 대비)</label><b class="{"hi" if vr >= 2 else ""}">{vr:.1f}배</b></div>'
        pf = ""
        if show_pf and r.get("qty"):
            pf = (
                f'<div class="pf"><span>평단 {money(r["avg"], cur)}</span>'
                f'<span class="{"up" if r["ret"] >= 0 else "down"}">{r["ret"]:+.1f}% ({money(r["pnl"], cur)})</span></div>'
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
  <div class="price"><b>{price_txt(r["close"], cur)}</b><span class="{"up" if r["chg"] >= 0 else "down"}">{r["chg"]:+.2f}%</span></div>
  {pf}
  <div class="why {"sb" if r["signal"]["buy_level"] else "ss" if r["signal"]["sell_level"] else ""}">{e(summary_line(r["signal"]))}</div>
  {r["spark"]}
  <div class="grid">
    <div><label>RSI(14)</label><b class="{rsi_cls(r["rsi"])}">{r["rsi"]:.1f}</b></div>
    <div><label>CCI(14)</label><b class="{cci_cls(r["cci"])}">{r["cci"]:.0f}</b></div>
    <div><label>볼린저 위치</label><b>{r["bb_pos"]:.0f}%</b></div>
    {vol_html}
  </div>
  <div class="mas">종가 위/아래: {ma_state}</div>
  {inv_html}
  {f'<div class="rtitle">💡 왜 이런 판단인지</div><ul class="reasons">{"".join(reasons)}</ul>' if reasons else ""}
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
<section id="{m["id"]}" class="pane"{"" if i == 0 else " hidden"}>
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
.why{{font-size:14px;background:var(--bg);border-radius:10px;padding:8px 10px;margin:8px 0}}.why.sb{{border-left:4px solid var(--buy)}}.why.ss{{border-left:4px solid var(--sell)}}li.buy{{color:var(--buy)}}li.sell{{color:var(--sell)}}
.pf{{display:flex;justify-content:space-between;font-size:14px}}details{{margin-top:8px;font-size:14px}}.news a{{color:inherit}}
footer{{font-size:12px;color:var(--mut);margin:20px 0}}
.tabs{{position:sticky;top:0;z-index:5;display:flex;gap:6px;background:var(--bg);padding:10px 0;margin-top:6px}}
.tab{{flex:1;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;padding:9px 0;font:inherit;font-weight:700;cursor:pointer}}
.pane{{scroll-margin-top:64px}}.tab.on{{background:var(--fg);color:var(--bg);border-color:var(--fg)}}.cnt{{display:inline-block;margin-left:5px;background:var(--sell);color:#fff;border-radius:99px;font-size:11px;padding:0 6px}}
</style></head><body><main>
<h1>내 주식 타이밍 보드</h1>
<div class="muted">{now:%Y-%m-%d %H:%M} 기준 · 10분마다 새로고침</div>
<nav class="tabs">{"".join(tabs)}</nav>
<div class="legend"><span><i style="color:var(--buy)">▲</i> 매수 타이밍</span><span><i style="color:var(--sell)">▼</i> 매도 타이밍</span><span>🔥 강한 매수</span><span>🐵 강한 매도</span>
<span><i style="color:#facc15">●</i> 골든크로스</span><span><i style="color:#64748b">●</i> 데드크로스</span>
<span><i style="color:#ec4899">―</i> 5일선</span><span><i style="color:#f59e0b">―</i> 20일선</span><span><i style="color:#8b5cf6">―</i> 60일선</span><span>┄ 볼린저</span></div>
{"".join(sections)}
<footer>그래프의 ▲▼는 지난 60일 동안 같은 기준(종목 자체 지표 2개 이상)을 매일 적용한 결과예요. 신호 기준: 같은 방향 지표가 2개 이상이면 신호, 3개 이상이면 강한 신호(매수 🔥, 매도 🐵).
참고용 기술적 신호이며 투자 판단은 본인 몫입니다.</footer>
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
  if (location.hash) show(location.hash.slice(1));
}})();
</script>
</body></html>"""




# ---------------------------------------------------------------- 실행

def load_portfolio() -> dict:
    raw = os.getenv("PORTFOLIO_JSON", "").strip()
    if not raw and (ROOT / "portfolio.json").exists():
        raw = (ROOT / "portfolio.json").read_text(encoding="utf-8")
    if not raw:
        return {}
    data = json.loads(raw)
    return {(str(k).zfill(6) if str(k).isdigit() else str(k).upper()): v for k, v in data.items()}


def analyze(stock: dict, df: pd.DataFrame, pf: dict, fg: dict | None, news: list[dict],
            inv: list[dict] | None = None, now: dt.datetime | None = None, market: dict | None = None) -> dict:
    market = market or {"id": "kr", "name": "국장", "cur": "KRW"}
    mid, cur = market["id"], market["cur"]
    ind = compute_indicators(df, CONFIG["indicators"])
    last, prev = ind.iloc[-1], ind.iloc[-2]
    now = now or dt.datetime.now(KST)
    scale = session_elapsed(now, mid) if ind.index[-1].date() == market_today(now, mid) else 1.0
    sig = evaluate(ind, fg, news_score(news, CONFIG["news"]), inv, scale)
    vol_ratio = (last["Volume"] / scale / last["VOL20"]) if last["VOL20"] else float("nan")
    width = last["BB_UP"] - last["BB_LO"]
    row = {
        **stock,
        "mid": mid,
        "market": market["name"],
        "cur": cur,
        "asof": ind.index[-1].strftime("%Y-%m-%d"),
        "close": float(last["Close"]),
        "chg": float((last["Close"] / prev["Close"] - 1) * 100),
        "rsi": float(last["RSI"]),
        "cci": float(last["CCI"]),
        "bb_pos": float((last["Close"] - last["BB_LO"]) / width * 100) if width else 50.0,
        "signal": sig,
        "news": news,
        "inv": inv or [],
        "vol_ratio": float(vol_ratio),
        "marks": (marks := history_marks(ind)),
        "spark": chart(ind, marks, cur=cur),
        "hist": recent_marks_html(marks, float(last["Close"]), cur=cur),
    }
    for n in CONFIG["indicators"]["ma"]:
        row[f"ma{n}"] = float(last[f"MA{n}"])
    h = pf.get(stock["code"])
    if h:
        qty, avg = float(h["qty"]), float(h["avg"])
        row.update(qty=qty, avg=avg, ret=(row["close"] / avg - 1) * 100, pnl=(row["close"] - avg) * qty)
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
        rows = []
        for s in m["stocks"]:
            try:
                df = fetch_ohlcv(s["code"], m["source"])
                if len(df) < 130:
                    raise ValueError(f"일봉 {len(df)}개뿐")
                q = fetch_quote(s["code"]) if m["id"] == "kr" else None
                inv = fetch_investor(s["code"]) if m["id"] == "kr" else None
                news = fetch_news(s.get("news") or s["name"], suffix=m.get("news_suffix", "주가"))
                row = analyze(s, apply_quote(df, q), pf, fg, news, inv, now, m)
                if q:  # 등락률은 증권사 앱과 같은 공식 값 사용
                    row["chg"] = q["chg"]
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {m['name']} {s['name']} 실패: {e}")
                rows.append({**s, "mid": m["id"], "error": str(e)[:80]})
        markets.append({**m, "rows": rows, "fg": fg, "idx": idx})

    # 알림: 같은 봉 날짜에 같은 신호는 1번만, 일반→강한 신호(🔥/🐵)로 올라가면 다시 알림
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
