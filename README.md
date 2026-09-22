# 비트코인 시계열 데이터 분석 (BTC-USD, 2023~2024)

시간에 따라 변화하는 시계열 데이터(비트코인 일별 가격)를 수집·정제·시각화하고,
트렌드/변동성/계절성을 분석하여 인사이트를 도출하는 프로젝트입니다.

## 폴더 구조

```
bitcoin-timeseries-analysis/
├── data/
│   └── btc_2023_2024.csv        # 수집된 원본 데이터 (실행 시 자동 생성)
├── images/
│   ├── 01_price_trend.png       # 가격 + 이동평균
│   ├── 02_daily_return.png      # 일간 수익률 + 롤링 변동성
│   ├── 03_monthly_return.png    # 월별 수익률
│   ├── 04_decomposition.png     # 시계열 분해 (보너스)
│   └── 05_forecast.png          # 베이스라인 예측 (보너스)
├── analysis.py                  # 메인 분석 스크립트
├── REPORT.md                    # 분석 리포트 (본 과제의 핵심 산출물)
├── requirements.txt             # 의존성 목록
└── README.md
```

## 실행 방법

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 분석 실행
python analysis.py
```

- 실행하면 데이터 수집 → 정제 → 지표 계산 → 시각화 → 시계열 분해/예측이
  순서대로 진행되고, `images/` 폴더에 그래프 5개가 저장됩니다.
- `data/btc_2023_2024.csv`가 이미 있으면 재사용하여 동일한 결과를 재현합니다.

## 데이터 출처 및 수집 방법

- **출처**: Yahoo Finance (`yfinance` 라이브러리, 티커 `BTC-USD`)
- **기간**: 2023-01-01 ~ 2024-12-30 (일별 730개 데이터 포인트)
- **수집 코드**:
  ```python
  import yfinance as yf
  df = yf.download("BTC-USD", start="2023-01-01", end="2024-12-31")
  ```
- **라이선스 주의**: Yahoo Finance 데이터는 개인 학습·연구 목적으로 사용하며,
  상업적 재배포 시에는 별도의 이용 약관 확인이 필요합니다.

## 개발 환경

- Python 3.10 이상
- 주요 라이브러리: pandas, matplotlib, numpy, statsmodels, yfinance

## 분석 리포트

전체 분석 내용(질문·데이터 설명·시각화·인사이트·결론·AI 사용 로그)은
[REPORT.md](REPORT.md) 에 정리되어 있습니다.
