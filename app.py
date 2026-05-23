import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import plotly.express as px
import ta
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Wall Street Financial Risk Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B0F19;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #121826;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("📊 Wall Street Financial Risk Dashboard")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Configuration")

    ticker_input = st.text_area(
        "Tickers",
        value="AAPL,MSFT,SPY"
    )

    tickers = [
        x.strip().upper()
        for x in ticker_input.split(",")
    ]

    benchmark = st.text_input(
        "Benchmark",
        value="^GSPC"
    )

    period = st.selectbox(
        "Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    )

    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo"]
    )

    risk_free_rate = st.slider(
        "Risk Free Rate",
        0.0,
        0.15,
        0.04
    )

# =========================================================
# DOWNLOAD DATA
# =========================================================

@st.cache_data
def download_data(tickers, period, interval):

    data = {}

    for ticker in tickers:

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False
        )

        if not df.empty:
            data[ticker] = df

    return data

market_data = download_data(
    tickers,
    period,
    interval
)

benchmark_data = yf.download(
    benchmark,
    period=period,
    interval=interval,
    progress=False
)

vix_data = yf.download(
    "^VIX",
    period=period,
    interval=interval,
    progress=False
)

# =========================================================
# FUNCTIONS
# =========================================================

def annualized_return(returns):
    return ((1 + returns.mean()) ** 252) - 1

def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)

def sharpe_ratio(ret, vol, rf):

    if vol == 0:
        return 0

    return (ret - rf) / vol

def beta(asset_returns, benchmark_returns):

    covariance = np.cov(asset_returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)

    return covariance / benchmark_variance

def value_at_risk(returns, confidence=0.95):

    mean = returns.mean()
    std = returns.std()

    return norm.ppf(
        1 - confidence,
        mean,
        std
    )

def add_indicators(df):

    df["EMA7"] = ta.trend.ema_indicator(df["Close"], window=7)
    df["EMA30"] = ta.trend.ema_indicator(df["Close"], window=30)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)

    bb = ta.volatility.BollingerBands(df["Close"])

    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_LOWER"] = bb.bollinger_lband()

    df["RSI"] = ta.momentum.rsi(df["Close"])

    macd = ta.trend.MACD(df["Close"])

    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    return df

def generate_signal(df):

    latest = df.iloc[-1]

    if latest["EMA7"] > latest["EMA30"] and latest["RSI"] < 70:
        return "BUY"

    elif latest["EMA7"] < latest["EMA30"] and latest["RSI"] > 30:
        return "SELL"

    return "HOLD"

def forecast_prices(df):

    model = ARIMA(df["Close"], order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=90)

    return forecast

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Benchmark",
    "Correlation",
    "Covariance",
    "Download"
])

# =========================================================
# OVERVIEW
# =========================================================

with tab1:

    summary = []

    for ticker in tickers:

        if ticker not in market_data:
            continue

        df = market_data[ticker].copy()

        df = add_indicators(df)

        returns = df["Close"].pct_change().dropna()

        benchmark_returns = benchmark_data["Close"].pct_change().dropna()

        min_len = min(len(returns), len(benchmark_returns))

        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]

        ann_return = annualized_return(returns)
        ann_vol = annualized_volatility(returns)

        sharpe = sharpe_ratio(
            ann_return,
            ann_vol,
            risk_free_rate
        )

        beta_value = beta(
            returns,
            benchmark_returns
        )

        var = value_at_risk(returns)

        signal = generate_signal(df)

        info = yf.Ticker(ticker).info

        pe_ratio = info.get("trailingPE")
        eps = info.get("trailingEps")
        market_cap = info.get("marketCap")
        free_cash_flow = info.get("freeCashflow")

        summary.append({
            "Ticker": ticker,
            "Annual Return": round(ann_return * 100, 2),
            "Volatility": round(ann_vol * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Beta": round(beta_value, 2),
            "VaR": round(var * 100, 2),
            "P/E": pe_ratio,
            "EPS": eps,
            "FCF": free_cash_flow,
            "Market Cap": market_cap,
            "Signal": signal
        })

        st.subheader(f"{ticker} Price Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA7"],
                name="EMA7"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA30"],
                name="EMA30"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(f"{ticker} Forecast")

        forecast = forecast_prices(df)

        forecast_df = pd.DataFrame({
            "Forecast": forecast
        })

        st.line_chart(forecast_df)

    summary_df = pd.DataFrame(summary)

    st.subheader("Portfolio Summary")

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    st.subheader("VIX Volatility Index")

    st.line_chart(vix_data["Close"])

# =========================================================
# BENCHMARK
# =========================================================

with tab2:

    fig = go.Figure()

    for ticker in tickers:

        if ticker not in market_data:
            continue

        temp = market_data[ticker]

        fig.add_trace(
            go.Scatter(
                x=temp.index,
                y=temp["Close"],
                name=ticker
            )
        )

    fig.add_trace(
        go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data["Close"],
            name=benchmark,
            line=dict(width=4)
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# CORRELATION
# =========================================================

with tab3:

    returns_df = pd.DataFrame()

    for ticker in tickers:

        if ticker not in market_data:
            continue

        returns_df[ticker] = (
            market_data[ticker]["Close"]
            .pct_change()
        )

    corr = returns_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu"
    )

    fig.update_layout(
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# COVARIANCE
# =========================================================

with tab4:

    cov = returns_df.cov() * 252

    fig = px.imshow(
        cov,
        text_auto=True,
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# DOWNLOAD
# =========================================================

with tab5:

    output_file = "financial_risk_analysis.xlsx"

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        corr.to_excel(
            writer,
            sheet_name="Correlation"
        )

        cov.to_excel(
            writer,
            sheet_name="Covariance"
        )

    with open(output_file, "rb") as f:

        st.download_button(
            "Download Excel",
            f,
            file_name=output_file
        )

st.success("Dashboard Loaded Successfully")