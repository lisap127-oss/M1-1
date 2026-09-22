"""
비트코인(BTC-USD) 시계열 데이터 분석 스크립트
=================================================
2023-01-01 ~ 2024-12-30 일별 가격 데이터를 수집하여
트렌드 / 변동성 / 계절성을 분석하고, 시계열 분해 및 간단 예측을 수행한다.

실행 방법:
    python analysis.py

산출물:
    - data/btc_2023_2024.csv        (원본 수집 데이터)
    - images/01_price_trend.png     (가격 + 이동평균)
    - images/02_daily_return.png    (일간 수익률 분포/변동성)
    - images/03_monthly_return.png  (월별 수익률 히트맵/막대)
    - images/04_decomposition.png   (시계열 분해, 보너스)
    - images/05_forecast.png        (베이스라인 예측, 보너스)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose

matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ------------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "btc_2023_2024.csv")


# ------------------------------------------------------------------
# 1. 데이터 수집
#    - yfinance 로 BTC-USD 일별 가격을 받아온다.
#    - 이미 CSV가 있으면 재사용하여 재현성을 확보한다.
# ------------------------------------------------------------------
def load_data():
    if os.path.exists(CSV_PATH):
        print(f"[1] 기존 CSV 로드: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)
        return df

    print("[1] yfinance 로 BTC-USD 데이터 다운로드 중...")
    import yfinance as yf
    raw = yf.download("BTC-USD", start="2023-01-01", end="2024-12-31", progress=False)
    # yfinance 신버전은 MultiIndex 컬럼을 반환하므로 평탄화한다.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.to_csv(CSV_PATH)
    print(f"    저장 완료: {CSV_PATH}")
    return df


# ------------------------------------------------------------------
# 2. 데이터 정제 및 기본 정보 확인
# ------------------------------------------------------------------
def clean_data(df):
    print("\n[2] 데이터 기본 정보")
    print(f"    기간: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"    행 수: {len(df)}")
    print(f"    컬럼: {list(df.columns)}")
    print("    결측치:\n", df.isnull().sum().to_string())

    # 결측치 처리: 가격 데이터는 직전 값으로 채움(전방 채움).
    # 암호화폐는 주말/공휴일 없이 거래되므로 실제로는 결측이 거의 없지만,
    # 방어적으로 처리 기준을 명시한다.
    if df.isnull().sum().sum() > 0:
        print("    -> 결측치 발견: 전방 채움(ffill)으로 처리")
        df = df.ffill()

    # 이상치 관찰용: 일간 수익률 기준 |z| > 3 인 날을 표시 (제거하지 않고 관찰만)
    ret = df["Close"].pct_change()
    z = (ret - ret.mean()) / ret.std()
    outliers = df.index[np.abs(z) > 3]
    print(f"    이상치(일간수익률 |z|>3) 후보 {len(outliers)}일 (제거하지 않고 관찰용으로 보존)")
    return df


# ------------------------------------------------------------------
# 3. 시계열 분석 지표 계산
#    - 이동평균(MA), 일간 수익률, 변동성, 월별 집계
# ------------------------------------------------------------------
def add_indicators(df):
    print("\n[3] 시계열 분석 지표 계산")
    df = df.copy()
    df["MA7"] = df["Close"].rolling(window=7).mean()      # 단기 이동평균
    df["MA30"] = df["Close"].rolling(window=30).mean()    # 장기 이동평균
    df["Return"] = df["Close"].pct_change()               # 일간 변화율
    df["Volatility30"] = df["Return"].rolling(window=30).std()  # 30일 롤링 변동성

    print(f"    전체 수익률: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.1f}%")
    print(f"    일간 변동성(표준편차): {df['Return'].std()*100:.2f}%")
    print(f"    연율화 변동성: {df['Return'].std()*np.sqrt(365)*100:.1f}%")
    return df


# ------------------------------------------------------------------
# 4. 시각화
# ------------------------------------------------------------------
def plot_price_trend(df):
    """시각화 1: 종가 + 이동평균선"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close", color="#F7931A", linewidth=1)
    plt.plot(df.index, df["MA7"], label="MA7 (short)", color="#1f77b4", linewidth=1)
    plt.plot(df.index, df["MA30"], label="MA30 (long)", color="#d62728", linewidth=1.2)
    plt.title("Bitcoin (BTC-USD) Price & Moving Averages, 2023-2024", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "01_price_trend.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    저장: {path}")


def plot_daily_return(df):
    """시각화 2: 일간 수익률 시계열 + 30일 롤링 변동성"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df.index, df["Return"] * 100, color="#555", linewidth=0.7)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_title("Daily Return (%)", fontsize=12)
    axes[0].set_ylabel("Return (%)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["Volatility30"] * 100, color="#9467bd", linewidth=1.2)
    axes[1].set_title("30-Day Rolling Volatility (Std of Daily Return, %)", fontsize=12)
    axes[1].set_ylabel("Volatility (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "02_daily_return.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    저장: {path}")


def plot_monthly_return(df):
    """시각화 3: 연-월별 수익률 막대그래프"""
    tmp = df.copy()
    tmp["YearMonth"] = tmp.index.to_period("M")
    monthly = tmp.groupby("YearMonth")["Close"].agg(["first", "last"])
    monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100

    plt.figure(figsize=(13, 6))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly["ret"]]
    plt.bar(monthly.index.astype(str), monthly["ret"], color=colors)
    plt.axhline(0, color="black", linewidth=0.6)
    plt.title("Monthly Return (%) by Year-Month", fontsize=13)
    plt.xlabel("Year-Month")
    plt.ylabel("Return (%)")
    plt.xticks(rotation=90)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "03_monthly_return.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    저장: {path}")


# ------------------------------------------------------------------
# 5. 보너스: 시계열 분해 + 간단 예측
# ------------------------------------------------------------------
def plot_decomposition(df):
    """보너스 A: 시계열 분해 (추세/계절성/잔차)"""
    # 30일 주기를 가정하여 분해 (월간 계절성 관찰 목적)
    result = seasonal_decompose(df["Close"], model="additive", period=30)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("Time Series Decomposition (period=30)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "04_decomposition.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    저장: {path}")


def plot_forecast(df):
    """보너스 B: 베이스라인 예측 (이동평균 기반 naive forecast)
    - 마지막 30일을 테스트 구간으로 두고, '직전 7일 평균'을 다음날 예측값으로 사용.
    - 정확도보다 '가정과 한계'를 설명하기 위한 베이스라인.
    """
    series = df["Close"].copy()
    horizon = 30
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    # naive baseline: 마지막 관측값을 그대로 미래로 연장(random walk 가정)
    last_value = train.iloc[-1]
    naive_pred = pd.Series([last_value] * horizon, index=test.index)

    # 이동평균 baseline: 직전 7일 평균을 다음날 예측으로 사용(rolling one-step)
    ma_pred_values = []
    history = list(train.values)
    for _ in range(horizon):
        ma_pred_values.append(np.mean(history[-7:]))
        history.append(np.mean(history[-7:]))  # 예측값을 다시 히스토리에 반영
    ma_pred = pd.Series(ma_pred_values, index=test.index)

    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-90:], train.iloc[-90:], label="Train (last 90d)", color="#333")
    plt.plot(test.index, test, label="Actual", color="#2ca02c", linewidth=1.8)
    plt.plot(test.index, naive_pred, label="Naive (last value)", color="#d62728", linestyle="--")
    plt.plot(test.index, ma_pred, label="MA7 baseline", color="#1f77b4", linestyle="--")
    plt.title("Baseline Forecast for Last 30 Days", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "05_forecast.png")
    plt.savefig(path, dpi=120)
    plt.close()

    # 오차 지표(MAE) 출력
    mae_naive = np.mean(np.abs(naive_pred.values - test.values))
    mae_ma = np.mean(np.abs(ma_pred.values - test.values))
    print(f"    저장: {path}")
    print(f"    Naive MAE: {mae_naive:.1f} USD / MA7 MAE: {mae_ma:.1f} USD")


# ------------------------------------------------------------------
# 메인 실행
# ------------------------------------------------------------------
def main():
    df = load_data()
    df = clean_data(df)
    df = add_indicators(df)

    print("\n[4] 시각화 생성")
    plot_price_trend(df)
    plot_daily_return(df)
    plot_monthly_return(df)

    print("\n[5] 보너스: 시계열 분해 & 예측")
    plot_decomposition(df)
    plot_forecast(df)

    print("\n완료. images/ 폴더의 그래프와 REPORT.md 를 확인하세요.")


if __name__ == "__main__":
    main()
