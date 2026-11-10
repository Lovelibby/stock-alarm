# 내 주식 타이밍 알람봇

평일 장중 30분마다 보유 종목의 **이동평균선 · 볼린저밴드 · RSI · CCI**를 실제 시세로 계산합니다.
**CNN 공포탐욕지수 · 코스피/코스닥 · 종목 뉴스**까지 보고, 매수·매도 신호가 뜨면 **텔레그램으로 알림**을 보냅니다.
결과는 휴대폰으로 보는 **웹 대시보드**에도 표시돼요.

- 비용: 무료 (GitHub Actions + GitHub Pages)
- PC 필요 없음: GitHub 서버가 대신 실행
- 처음 설정: 15분 정도 (휴대폰으로 가능, 아래 순서대로)

---

## 1단계. 텔레그램 봇 만들기 (5분)

1. 텔레그램에서 **@BotFather** 검색 → 대화 시작
2. `/newbot` 입력 → 봇 이름(예: `내주식알람`) → 봇 아이디(예: `my_stock_alarm_123_bot`, 끝이 `bot`)
3. BotFather가 주는 **토큰**(`123456789:AAH...` 형태)을 복사해서 메모 → 이게 `TELEGRAM_BOT_TOKEN`
4. 방금 만든 봇을 검색해서 들어가 **아무 메시지나 1개** 보내기 (예: `안녕`)
5. 휴대폰 브라우저 주소창에 아래 주소 입력 (`토큰` 부분을 3번 값으로 바꿔서)
   ```
   https://api.telegram.org/bot토큰/getUpdates
   ```
6. 화면에서 `"chat":{"id":` 뒤의 숫자(예: `987654321`)를 메모 → 이게 `TELEGRAM_CHAT_ID`

## 2단계. GitHub에 올리기 (7분)

> 휴대폰 브라우저에서 하면 메뉴가 숨어 있을 수 있어요. 브라우저 메뉴에서 **'데스크톱 사이트'**를 켜면 편해요.

1. [github.com](https://github.com) 가입 → 오른쪽 위 **+ → New repository**
2. 이름 `stock-alarm`, **Public** 선택 → **Create repository**
   - Public이어야 대시보드(GitHub Pages)를 무료로 쓸 수 있어요.
   - 내 보유 수량·평단은 파일에 넣지 않고 비밀값(Secret)으로 넣기 때문에 공개되지 않아요.
3. **Add file → Upload files** → 압축 푼 폴더의 `alarm.py`, `config.json`, `requirements.txt`, `README.md` 올리기 → **Commit changes**
4. **Add file → Create new file**
   - 파일 이름 칸에 정확히 `.github/workflows/alarm.yml` 입력 (슬래시를 치면 폴더가 자동으로 생겨요)
   - 내용 칸에 `alarm.yml` 파일 내용을 전부 복사해 붙여넣기 → **Commit changes**

## 3단계. 비밀값 넣기 (3분)

저장소에서 **Settings → Secrets and variables → Actions → New repository secret**을 눌러 3개를 만드세요.

| Name | Secret (값) |
|---|---|
| `TELEGRAM_BOT_TOKEN` | 1단계 3번 토큰 |
| `TELEGRAM_CHAT_ID` | 1단계 6번 숫자 |
| `PORTFOLIO_JSON` | 아래 내용 그대로 (수량·평단 바뀌면 여기만 수정) |

```json
{"018260":{"qty":18,"avg":181864},"005930":{"qty":11,"avg":285505},"017670":{"qty":5,"avg":125051},"005380":{"qty":16,"avg":480408},"012330":{"qty":10,"avg":467462},"204320":{"qty":57,"avg":54104},"058610":{"qty":35,"avg":121057},"108490":{"qty":1,"avg":276984}}
```

## 4단계. 켜기 (2분)

1. **Settings → Actions → General** → 맨 아래 *Workflow permissions*에서 **Read and write permissions** 선택 → Save
2. **Actions** 탭 → 왼쪽 `주식 타이밍 알람` → **Run workflow** → 1~2분 뒤 초록 체크가 뜨면 성공 (이때 `docs` 폴더가 생겨요)
3. **Settings → Pages** → Source: **Deploy from a branch**, Branch: **main**, 폴더: **/docs** → Save
4. 몇 분 뒤 대시보드 주소: `https://내아이디.github.io/stock-alarm/` (홈 화면에 추가하면 앱처럼 쓸 수 있어요)

---

## 언제 알림이 오나요?

같은 방향 지표가 **2개 이상** 겹치면 알림이 오고, **3개 이상**이면 🔥 강한 신호로 표시돼요.
단, 종목 자체 기술적 지표(아래 표 위쪽 4줄)가 최소 1개는 있어야 해요. 시장이 공포라는 이유만으로 전 종목에 알림이 쏟아지는 걸 막기 위해서예요.

| 지표 | 매수 쪽 | 매도 쪽 |
|---|---|---|
| RSI(14) | 30 이하 | 70 이상 |
| CCI(14) | -100 이하 | +100 이상 |
| 볼린저밴드(20,2) | 하단 터치 | 상단 터치 |
| 이동평균선 | 5일선이 20일선 상향 돌파 / 20일선 회복 | 5일선이 20일선 하향 돌파 / 20일선 이탈 |
| 공포탐욕(CNN) | 25 이하 (극단적 공포) | 75 이상 (극단적 탐욕) |
| 뉴스(최근 2일) | 호재성 제목이 2건 이상 많음 | 악재성 제목이 2건 이상 많음 |

- 같은 날 같은 신호는 **한 번만** 와요. 일반 신호가 🔥로 강해지면 한 번 더 와요.
- 실행 시각은 평일 09:00~15:30 30분마다, 그리고 15:50 마감 확정값이에요. GitHub 사정으로 5~15분 늦을 수 있어요.

## 바꾸고 싶을 때

- **종목 추가/삭제**: `config.json`의 `stocks` 목록 수정, `PORTFOLIO_JSON` 비밀값 수정
- **기준 변경** (예: RSI 35 이하로 더 민감하게): `config.json`의 `thresholds`
- **대시보드에 평단·손익도 표시**: Settings → Secrets and variables → Actions → **Variables** 탭 → `SHOW_PORTFOLIO` = `true`
  (대시보드는 공개 주소라서, 켜면 남들도 내 평단을 볼 수 있어요)

## 참고

- 시세는 FinanceDataReader(네이버 금융 데이터)로 받아요. 장중에는 그날 봉이 진행 중인 값이에요.
- 저장소에 60일 넘게 아무 변화가 없으면 GitHub이 자동 실행을 멈출 수 있어요. 이 봇은 실행할 때마다 대시보드를 갱신하며 커밋하니 보통 괜찮지만, 멈추면 Actions 탭에서 다시 켜면 돼요.
- 참고용 기술적 신호예요. 투자 판단은 본인 몫입니다.
