"""차트 기반 점수 엔진 (Libby Stock Score — 기술적 분석 부분).

화면·알림과 분리된 순수 계산 모듈:
  - 지지선·저항선 자동 계산
  - 매수 타이밍 100점 (중장기 추세 25, 모멘텀 20, 거래량 15, 수급 15, 지지·저항 15, 시장 심리 10)
    → 기획서의 40점 배분(10·8·6·6·6·4)을 같은 비율로 100점에 맞춤 (×2.5)
  - 매도 위험 100점 중 기술적 60점 (추세 약화 20, 과열·둔화 15, 거래량·수급 15, 시장 위험 10)
    → 밸류에이션 20, 펀더멘털 20은 재무 데이터 연결 전까지 N/A
  - 분할매수 구간, 손절 고려가, 1·2차 목표가

원칙: 데이터가 없으면 0점이 아니라 N/A. N/A를 뺀 나머지로 환산하고 신뢰도를 함께 표시.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 추가 지표

def add_indicators(ind: pd.DataFrame) -> pd.DataFrame:
    """MACD(12,26,9), ATR(14), 거래대금, 120일선을 추가."""
    out = ind
    c, h, l = out["Close"], out["High"], out["Low"]
    if "MA120" not in out:
        out["MA120"] = c.rolling(120).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIG"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIG"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    out["VALUE"] = c * out["Volume"]
    return out


# ---------------------------------------------------------------- 지지·저항

def pivots(ind: pd.DataFrame, k: int = 3, lookback: int = 120) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """좌우 k일보다 높은 고점 / 낮은 저점 (확정된 것만, 최근 lookback일)."""
    d = ind.iloc[-lookback:]
    hi, lo = d["High"].values, d["Low"].values
    base = len(ind) - len(d)
    highs, lows = [], []
    for i in range(k, len(d) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            highs.append((base + i, float(hi[i])))
        if lo[i] == lo[i - k:i + k + 1].min():
            lows.append((base + i, float(lo[i])))
    return highs, lows


def _cluster(levels: list[float], tol: float = 0.015) -> list[float]:
    """1.5% 안쪽으로 붙은 가격대는 하나로 묶음."""
    out: list[list[float]] = []
    for v in sorted(levels):
        if out and abs(v / np.mean(out[-1]) - 1) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out]


def levels(ind: pd.DataFrame) -> dict:
    """현재가 기준 가까운 지지선 2개, 저항선 2개."""
    close = float(ind["Close"].iloc[-1])
    highs, lows = pivots(ind)
    sup = sorted([v for v in _cluster([p for _, p in lows]) if v < close * 0.995], reverse=True)[:2]
    res = sorted([v for v in _cluster([p for _, p in highs]) if v > close * 1.005])[:2]
    return {"support": sup, "resistance": res, "highs": highs, "lows": lows}


# ---------------------------------------------------------------- 공통 도우미

def _item(name: str, mx: int, score: float | None, reason: str) -> dict:
    if score is not None:
        score = float(max(0, min(mx, round(score, 1))))
    return {"name": name, "max": mx, "score": score, "reason": reason}


def _total(items: list[dict], scale: int) -> dict:
    avail = [i for i in items if i["score"] is not None]
    got, mx = sum(i["score"] for i in avail), sum(i["max"] for i in avail)
    full = sum(i["max"] for i in items)
    value = round(got / mx * scale, 1) if mx else None
    return {"value": value, "scale": scale, "raw": got, "avail_max": mx, "full_max": full,
            "confidence": round(mx / full * 100) if full else 0}


def _vol_ratio(ind: pd.DataFrame, i: int, scale: float) -> float | None:
    v20 = ind["Volume"].iloc[max(0, i - 20):i].mean()
    v = ind["Volume"].iloc[i] / (scale or 1)
    return float(v / v20) if v20 and v > 0 else None


def _flow(inv: list[dict] | None) -> tuple[float, int, int] | None:
    """외국인+기관 최근 5일: (합계, 순매수일 수, 순매도일 수)."""
    if not inv:
        return None
    net = [d["foreign"] + d["inst"] for d in inv[:5]]
    return sum(net), sum(1 for x in net if x > 0), sum(1 for x in net if x < 0)


def _value_trend(ind: pd.DataFrame) -> float | None:
    """최근 5일 평균 거래대금 / 60일 평균 거래대금."""
    v = ind["VALUE"]
    a, b = v.iloc[-6:-1].mean(), v.iloc[-61:-1].mean()
    return float(a / b) if b and not np.isnan(a) else None


def _lower_structure(lv: dict) -> bool:
    """최근 고점 2개와 저점 2개가 모두 낮아지는 중 (하락 추세 구조)."""
    h, l = lv["highs"][-2:], lv["lows"][-2:]
    return len(h) == 2 and len(l) == 2 and h[1][1] < h[0][1] and l[1][1] < l[0][1]


def _bear_divergence(ind: pd.DataFrame) -> bool:
    """최근 20일 신고가인데 RSI는 이전 고점 때보다 낮음 (상승 힘 약화)."""
    if len(ind) < 60:
        return False
    c, r = ind["Close"], ind["RSI"]
    if c.iloc[-1] < c.iloc[-20:].max() * 0.99:
        return False
    prev = c.iloc[-60:-20]
    j = prev.idxmax()
    return bool(c.iloc[-1] >= prev.max() and r.iloc[-1] < r.loc[j] - 3)


# ---------------------------------------------------------------- 매수 타이밍 100점

def timing_score(ind: pd.DataFrame, lv: dict, fg: dict | None, inv: list[dict] | None,
                 mid: str = "kr", vol_scale: float = 1.0) -> dict:
    last, prev = ind.iloc[-1], ind.iloc[-2]
    close = float(last["Close"])
    items = []

    # ① 중장기 추세 10
    if np.isnan(last["MA60"]):
        items.append(_item("중장기 추세", 10, None, "60일 이상 데이터가 없어 판단 보류"))
    else:
        s, why = 0.0, []
        if close > last["MA20"]:
            s += 3; why.append("주가가 20일선 위")
        else:
            why.append("주가가 20일선 아래")
        if last["MA20"] > last["MA60"]:
            s += 3; why.append("20일선이 60일선 위(중기 상승 배열)")
        if not np.isnan(last["MA120"]):
            if last["MA60"] > last["MA120"]:
                s += 2; why.append("60일선이 120일선 위(장기도 상승)")
        slope = last["MA60"] / ind["MA60"].iloc[-11] - 1 if len(ind) > 11 else 0
        if slope > 0:
            s += 2; why.append("60일선이 위로 기울어짐")
        else:
            why.append("60일선이 아래로 기울어짐")
        items.append(_item("중장기 추세", 10, s, " · ".join(why)))

    # ② 모멘텀 8 (RSI + MACD를 한 항목으로)
    rsi, rsi_up = float(last["RSI"]), last["RSI"] > prev["RSI"]
    if rsi >= 70:
        rs, rw = 0, f"RSI {rsi:.0f}: 과열이라 지금 사기엔 늦을 수 있음"
    elif rsi >= 60:
        rs, rw = 2, f"RSI {rsi:.0f}: 꽤 오른 상태"
    elif rsi >= 40:
        rs, rw = (4, f"RSI {rsi:.0f}: 건강한 구간에서 상승 중") if rsi_up else (3, f"RSI {rsi:.0f}: 보통")
    elif rsi >= 30:
        rs, rw = (4, f"RSI {rsi:.0f}: 바닥권에서 고개를 드는 중") if rsi_up else (2, f"RSI {rsi:.0f}: 약세 지속")
    else:
        rs, rw = (3, f"RSI {rsi:.0f}: 과매도에서 반등 시작") if rsi_up else (1, f"RSI {rsi:.0f}: 과매도지만 아직 하락 중")
    ms, mw = 0, []
    if last["MACD"] > last["MACD_SIG"]:
        ms += 2; mw.append("MACD가 시그널 위(상승 힘 우세)")
    else:
        mw.append("MACD가 시그널 아래")
    if last["MACD_HIST"] > prev["MACD_HIST"]:
        ms += 2; mw.append("상승 힘이 커지는 중")
    items.append(_item("모멘텀", 8, rs + ms, rw + " · " + " · ".join(mw)))

    # ③ 거래량 6
    vr = _vol_ratio(ind, len(ind) - 1, vol_scale)
    up = close >= float(prev["Close"])
    if vr is None:
        items.append(_item("거래량", 6, None, "거래량 데이터 없음"))
    elif up and vr >= 1.5:
        items.append(_item("거래량", 6, 6, f"평소의 {vr:.1f}배 거래량이 실린 상승 → 진짜 힘일 가능성"))
    elif up and vr >= 1.0:
        items.append(_item("거래량", 6, 4, f"평소 수준({vr:.1f}배) 거래와 함께 상승"))
    elif not up and vr >= 1.5:
        items.append(_item("거래량", 6, 0, f"평소의 {vr:.1f}배 물량이 쏟아지며 하락 → 매도세 강함"))
    elif up:
        items.append(_item("거래량", 6, 2, f"거래가 적은({vr:.1f}배) 상승 → 힘이 약할 수 있음"))
    else:
        items.append(_item("거래량", 6, 3, f"거래가 크지 않은 하락({vr:.1f}배) → 큰 매도는 아님"))

    # ④ 수급 6 (국장: 외국인·기관 / 그 외: 거래대금 추세)
    fl = _flow(inv) if mid == "kr" else None
    if fl:
        tot, pos, neg = fl
        if tot > 0 and pos >= 3:
            items.append(_item("수급", 6, 6, f"외국인·기관이 최근 5일 중 {pos}일 순매수"))
        elif tot > 0:
            items.append(_item("수급", 6, 4, "외국인·기관 5일 합계 순매수"))
        elif tot < 0 and neg >= 3:
            items.append(_item("수급", 6, 0, f"외국인·기관이 최근 5일 중 {neg}일 순매도"))
        else:
            items.append(_item("수급", 6, 2, "외국인·기관 5일 합계 순매도"))
    elif mid == "kr":
        items.append(_item("수급", 6, None, "외국인·기관 데이터를 받지 못함"))
    else:
        vt = _value_trend(ind)
        if vt is None:
            items.append(_item("수급(거래대금)", 6, None, "거래대금 데이터 부족"))
        else:
            s = 5 if vt >= 1.3 and close > last["MA20"] else 4 if vt >= 1.0 else 3 if vt >= 0.8 else 1
            items.append(_item("수급(거래대금)", 6, s, f"최근 거래대금이 평소의 {vt:.1f}배 (돈이 {'몰리는' if vt >= 1 else '빠지는'} 중)"))

    # ⑤ 지지·저항 6
    sup, res = lv["support"], lv["resistance"]
    if not sup:
        items.append(_item("지지·저항", 6, None, "가까운 지지선을 찾지 못함 (신저가 구간 등)"))
    else:
        ds = close / sup[0] - 1
        s, why = (2.0 if ds <= 0.15 else 0.0), [f"가까운 지지선 {sup[0]:,.0f}까지 -{ds * 100:.1f}%"]
        if ds > 0.15:
            why.append("지지선이 너무 멀어 손절 기준을 잡기 어려움")
        if ds <= 0.05:
            s += 2; why.append("지지선 근처라 손실 위험이 작음")
        if res:
            dr = res[0] / close - 1
            if ds > 0 and dr / max(ds, 0.005) >= 2:
                s += 1; why.append(f"저항선까지 +{dr * 100:.1f}% 여유 (기대 수익 > 위험)")
        else:
            s += 1; why.append("위쪽에 뚜렷한 저항선 없음")
        reentry = any(ind["Close"].iloc[j - 1] < ind["BB_LO"].iloc[j - 1] and ind["Close"].iloc[j] > ind["BB_LO"].iloc[j]
                      for j in range(len(ind) - 3, len(ind)))
        if reentry:
            s += 1; why.append("볼린저 하단 이탈 후 밴드 안으로 재진입 (반전 신호)")
        items.append(_item("지지·저항", 6, s, " · ".join(why)))

    # ⑥ 시장 심리 4 (공포일수록 역발상 매수 기회)
    if not fg:
        items.append(_item("시장 심리", 4, None, "공포탐욕지수를 받지 못함"))
    else:
        v = fg["score"]
        s = 4 if v <= 25 else 3 if v <= 45 else 2 if v <= 55 else 1 if v < 75 else 0
        mood = "극단적 공포 → 싸게 살 기회가 많은 구간" if v <= 25 else "공포" if v <= 45 else "중립" if v <= 55 \
            else "탐욕" if v < 75 else "극단적 탐욕 → 고점 주의"
        items.append(_item("시장 심리", 4, s, f"{fg.get('label', '공포탐욕')} {v:.0f} ({mood})"))

    # 항목 안에서는 기획서의 40점 배분으로 계산한 뒤, 같은 비율로 100점 만점에 맞춤
    for it in items:
        it["max"] = it["max"] * 2.5
        if it["score"] is not None:
            it["score"] = round(it["score"] * 2.5, 1)
        it["max"] = int(it["max"]) if float(it["max"]).is_integer() else it["max"]
    total = _total(items, 100)
    v = total["value"]
    total["grade"] = None if v is None else ("매우 좋음" if v >= 85 else "좋음" if v >= 75 else "관심"
                                            if v >= 65 else "관망" if v >= 50 else "나쁨")
    return {"items": items, **total}


# ---------------------------------------------------------------- 매도 위험 100점 (기술 60 + 재무 40 보류)

def sell_risk(ind: pd.DataFrame, lv: dict, fg: dict | None, inv: list[dict] | None,
              idx_chg: float | None = None, mid: str = "kr", vol_scale: float = 1.0) -> dict:
    last, prev = ind.iloc[-1], ind.iloc[-2]
    close = float(last["Close"])
    items = []

    # ① 추세 약화 20
    s, why = 0.0, []
    if close < last["MA20"]:
        s += 6; why.append("20일선 아래로 내려옴")
    if not np.isnan(last["MA60"]) and close < last["MA60"]:
        s += 6; why.append("60일선 아래로 내려옴")
    if not np.isnan(last["MA60"]) and last["MA20"] < last["MA60"]:
        s += 4; why.append("20일선이 60일선 아래(중기 하락 배열)")
    if last["MACD"] < last["MACD_SIG"]:
        s += 2; why.append("MACD 하락 전환")
    if _lower_structure(lv):
        s += 2; why.append("고점·저점이 계속 낮아지는 중")
    items.append(_item("추세 약화", 20, s, " · ".join(why) or "추세가 아직 살아 있음"))

    # ② 과열·둔화 15
    s, why = 0.0, []
    rsi = float(last["RSI"])
    if rsi >= 80:
        s += 10; why.append(f"RSI {rsi:.0f} 매우 과열")
    elif rsi >= 70:
        s += 7; why.append(f"RSI {rsi:.0f} 과열")
    if _bear_divergence(ind):
        s += 4; why.append("가격은 신고가인데 RSI는 낮아짐 (상승 힘 약화)")
    gap = close / last["MA20"] - 1
    if gap >= 0.25:
        s += 5; why.append(f"20일선보다 {gap * 100:.0f}% 위 (너무 멀리 올라감)")
    elif gap >= 0.15:
        s += 3; why.append(f"20일선보다 {gap * 100:.0f}% 위")
    items.append(_item("과열·둔화", 15, s, " · ".join(why) or "과열 신호 없음"))

    # ③ 거래량·수급 15
    s, why = 0.0, []
    vr = _vol_ratio(ind, len(ind) - 1, vol_scale)
    up = close >= float(prev["Close"])
    if vr is not None and not up and vr >= 1.5:
        s += 6; why.append(f"평소의 {vr:.1f}배 거래량을 동반한 하락")
    if vr is not None and up and vr < 0.7 and gap > 0.05:
        s += 4; why.append("오르긴 하는데 거래가 줄어듦 (상승 탄력 둔화)")
    fl = _flow(inv) if mid == "kr" else None
    flow_na = mid == "kr" and fl is None
    if fl:
        tot, pos, neg = fl
        if tot < 0 and neg >= 3:
            s += 9; why.append(f"외국인·기관이 5일 중 {neg}일 순매도")
        elif tot < 0:
            s += 5; why.append("외국인·기관 5일 합계 순매도")
    elif mid != "kr":
        vt = _value_trend(ind)
        if vt is not None and vt < 0.8 and up:
            s += 3; why.append("거래대금이 줄어드는 상승")
    reason = " · ".join(why) or ("매도 압력 뚜렷하지 않음" + (" (수급 데이터 없음)" if flow_na else ""))
    items.append(_item("거래량·수급", 15, s, reason))

    # ④ 시장 위험 10
    s, why = 0.0, []
    if fg:
        if fg["score"] >= 85:
            s += 6; why.append(f"{fg.get('label', '공포탐욕')} {fg['score']:.0f} 극단적 탐욕")
        elif fg["score"] >= 75:
            s += 4; why.append(f"{fg.get('label', '공포탐욕')} {fg['score']:.0f} 탐욕 과열")
    if idx_chg is not None and idx_chg <= -2:
        s += 2; why.append(f"시장 지수 {idx_chg:+.1f}% 급락")
    atr_pct = (ind["ATR"] / ind["Close"])
    if len(atr_pct.dropna()) > 60 and atr_pct.iloc[-1] > atr_pct.iloc[-120:].median() * 1.5:
        s += 2; why.append("평소보다 변동성이 크게 커짐")
    items.append(_item("시장 위험", 10, s if (fg or idx_chg is not None) else None,
                       " · ".join(why) or "시장 위험 신호 없음"))

    # ⑤⑥ 재무 (보류)
    items.append(_item("밸류에이션 부담", 20, None, "재무 데이터 연결 전이라 보류"))
    items.append(_item("펀더멘털 악화", 20, None, "재무 데이터 연결 전이라 보류"))

    total = _total(items, 100)
    # 시장 위험은 '가산점'으로만 반영: 시장이 조용하다고 종목 자체 위험이 희석되지 않게
    mkt = items[3]
    if total["value"] is not None and mkt["score"] is not None:
        base = total["avail_max"] - mkt["max"]
        total["value"] = round(min(100.0, total["raw"] / base * 100), 1) if base else None
        total["note"] = "시장 위험은 가산점으로만 반영"
    v = total["value"]
    total["grade"] = None if v is None else ("강한 매도 경고" if v >= 80 else "매도 주의" if v >= 65 else
                                            "보유 주의" if v >= 45 else "정상 보유" if v >= 25 else "매도 위험 낮음")
    return {"items": items, **total}


# ---------------------------------------------------------------- 가격 계획

def price_plan(ind: pd.DataFrame, lv: dict) -> dict:
    close = float(ind["Close"].iloc[-1])
    atr = float(ind["ATR"].iloc[-1]) if not np.isnan(ind["ATR"].iloc[-1]) else close * 0.03
    sup, res = lv["support"], lv["resistance"]
    plan = {"support": sup, "resistance": res, "atr": atr}
    if sup:
        plan["buy_zone"] = [sup[0], min(sup[0] * 1.02, close)]
        plan["buy_zone2"] = sup[1] if len(sup) > 1 else None
        plan["stop"] = sup[0] - atr  # 지지선이 깨지고 하루 평균 변동폭만큼 더 빠지면
    else:
        plan["buy_zone"], plan["buy_zone2"], plan["stop"] = None, None, close - 2 * atr
    t1 = res[0] if res else close + 2 * atr
    t2 = res[1] if len(res) > 1 else max(t1, close) + 2 * atr
    plan.update(target1=t1, target2=t2, target_basis="저항선" if res else "변동성(ATR) 기준")
    return plan


# ---------------------------------------------------------------- 과거 신호 (차트 표시용)

def history(ind: pd.DataFrame, n: int, mid: str) -> list[dict]:
    """최근 n일 동안 매일 같은 기준(그날까지의 데이터만)으로 점수를 다시 계산.
    과거 공포탐욕·수급 데이터는 없어서 차트 지표만으로 계산 (N/A 항목 제외 환산)."""
    marks, was_b, was_s = [], 0, 0
    start = max(130, len(ind) - n)
    for i in range(start, len(ind)):
        sub = ind.iloc[: i + 1]
        lv = levels(sub)
        t = timing_score(sub, lv, None, None, mid="x")
        r = sell_risk(sub, lv, None, None, None, mid="x")
        tv, rv = t["value"] or 0, r["value"] or 0
        b = 2 if tv >= 85 else 1 if tv >= 75 else 0
        s = 2 if rv >= 80 else 1 if rv >= 65 else 0
        day, px = ind.index[i].strftime("%m/%d"), float(ind["Close"].iloc[i])
        if b > was_b:
            marks.append({"i": i, "kind": "buy", "strong": b == 2, "day": day, "price": px,
                          "label": f"매수 타이밍 {tv:.0f}/100"})
        if s > was_s:
            marks.append({"i": i, "kind": "sell", "strong": s == 2, "day": day, "price": px,
                          "label": f"매도 위험 {rv:.0f}/100"})
        was_b, was_s = b, s
    return marks
