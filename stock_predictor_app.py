import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime

# --- Page Config ---
st.set_page_config(
    page_title="Finance Flashback",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Dark Neon Theme ---
st.markdown(
    """
    <style>
    body, .stApp { background: #18122B !important; color: #F6F6F6 !important; font-family: 'Montserrat', sans-serif; }
    .stButton>button {
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        color: #18122B !important;
        border-radius: 8px;
        font-weight: 700;
        border: none;
        box-shadow: 0 0 10px #00F2FE, 0 0 20px #4FACFE;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #18122B !important;
        box-shadow: 0 0 20px #43e97b, 0 0 40px #38f9d7;
    }
    .stSidebar { background: #231942 !important; }
    .stTextInput>div>div>input {
        background: #231942 !important;
        color: #F6F6F6 !important;
        border: 2px solid #00F2FE !important;
        border-radius: 8px;
        font-size: 1.1em;
        font-family: 'Montserrat', monospace;
        box-shadow: 0 0 10px #00F2FE;
        caret-color: #00F2FE;
    }
    .stDataFrame, .stTable { background: #231942 !important; color: #F6F6F6 !important; }
    .css-1v0mbdj { background: #231942 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #00F2FE !important;
        text-shadow: 0 0 10px #00F2FE;
    }
    </style>
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# --- Welcome Message & GIF ---
st.markdown("""
# Welcome to Finance Flashback 💸

<div style='text-align:center;'>
    <img src='https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXBheGQ4NnV1cXNkbG1oaTZvZTRpcGMyNG04MjgyOG82OGFrYzA2NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/se5RgFVR3GgfjynNlb/giphy.gif' width='300'/>
</div>

""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("📊 Data Options")
kragle_file = st.sidebar.file_uploader("Upload Kragle Dataset (CSV)", type=["csv"])

st.sidebar.markdown("---")

stock_ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="",
    placeholder="e.g. TSLA, AAPL, MSFT",
    help="Type a stock symbol to fetch real-time data."
)
today = datetime.date.today()
default_start = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)
run_pred = st.sidebar.button("Predict with Linear Regression")

# --- Neon Prompt ---
st.markdown(
    """
    <div style='margin-top: 2em; margin-bottom: 2em; text-align:center;'>
        <span style='font-size:1.3em; color:#00F2FE; text-shadow:0 0 10px #00F2FE;'>
            <span style='border-right: .15em solid #00F2FE; padding-right:0.2em; animation: blink-cursor 1s steps(1) infinite;'>
                Ready to uncover the trends? Type a stock symbol or upload a dataset to begin...
            </span>
        </span>
    </div>
    <style>
    @keyframes blink-cursor {
      0% { border-color: #00F2FE; }
      49% { border-color: #00F2FE; }
      50% { border-color: transparent; }
      100% { border-color: transparent; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Logic ---
stock_data = None
show_forecast = False

def plot_stock(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='#00F2FE')))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        plot_bgcolor='#231942',
        paper_bgcolor='#231942',
        font=dict(color='#F6F6F6')
    )
    st.plotly_chart(fig, use_container_width=True)

def linear_regression_forecast(df, days=30):
    df = df.copy()
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_ordinal'] = df['Date'].map(lambda x: x.toordinal())
    X = df['Date_ordinal'].values.reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    forecast = model.predict(future_ordinals)
    forecast_df = pd.DataFrame({
        'Date': [d.date() for d in future_dates],
        'Forecast': forecast.flatten()
    })
    return forecast_df

def plot_forecast(df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical', line=dict(color='#00F2FE')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='#4FACFE', dash='dash')))
    fig.update_layout(
        title='30-Day Linear Regression Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        plot_bgcolor='#231942',
        paper_bgcolor='#231942',
        font=dict(color='#F6F6F6')
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Data Handling ---
if kragle_file is not None:
    st.subheader("🔍 Kragle Dataset Preview")
    kragle_df = pd.read_csv(kragle_file)
    st.dataframe(kragle_df.head())
    st.button("Run Kragle Analysis 🚀 (Coming Soon)")
elif run_pred and stock_ticker:
    with st.spinner(f"Fetching data for {stock_ticker.upper()}..."):
        try:
            df = yf.download(stock_ticker.upper(), start=start_date, end=end_date)
            if df.empty or 'Close' not in df.columns:
                st.warning("No data found or missing 'Close' prices for this ticker.")
            else:
                df = df[['Close']].dropna()
                st.subheader(f"Historical Closing Prices for {stock_ticker.upper()}")
                hist_df = df.reset_index().copy()
                hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.strftime('%Y-%m-%d')
                hist_df = hist_df[['Date', 'Close']]
                st.dataframe(hist_df.tail(100), use_container_width=True)

                # Prepare data for regression (last 30 days)
                if len(df) < 31:
                    st.warning("Not enough data for 30-day prediction.")
                else:
                    df_last = df[-31:].copy()
                    df_last = df_last.reset_index()
                    df_last['Date_ordinal'] = df_last['Date'].map(lambda x: x.toordinal())
                    X = df_last['Date_ordinal'][:-1].values.reshape(-1, 1)
                    y = df_last['Close'][:-1].values
                    X_pred = df_last['Date_ordinal'][1:].values.reshape(-1, 1)

                    # Linear Regression
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X_pred)

                    # Neon/3D style matplotlib plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#18122B')
                    ax.set_facecolor('#231942')

                    # Neon glow for actual close
                    for glow in range(10, 0, -2):
                        ax.plot(
                            df_last['Date'][1:], df_last['Close'][1:],
                            color='#00F2FE', linewidth=glow/2, alpha=0.05*glow, zorder=1
                        )
                    ax.plot(
                        df_last['Date'][1:], df_last['Close'][1:],
                        label='Actual Close', color='#00F2FE', marker='o', linewidth=2.5, zorder=2
                    )

                    # Neon glow for predicted close
                    for glow in range(10, 0, -2):
                        ax.plot(
                            df_last['Date'][1:], y_pred,
                            color='#43e97b', linewidth=glow/2, alpha=0.05*glow, linestyle='--', zorder=1
                        )
                    ax.plot(
                        df_last['Date'][1:], y_pred,
                        label='Predicted Close', color='#43e97b', linestyle='--', marker='x', linewidth=2.5, zorder=2
                    )

                    ax.set_title(f"{stock_ticker.upper()} - Actual vs Predicted Closing Prices (Last 30 Days)", color='#00F2FE', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Date', color='#F6F6F6', fontsize=12)
                    ax.set_ylabel('Price (USD)', color='#F6F6F6', fontsize=12)
                    ax.legend(facecolor='#231942', edgecolor='#00F2FE', fontsize=12)
                    ax.grid(True, alpha=0.2, color='#00F2FE')
                    plt.xticks(rotation=30, color='#F6F6F6')
                    plt.yticks(color='#F6F6F6')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#00F2FE')
                        spine.set_linewidth(1.5)
                    fig.tight_layout()
                    st.pyplot(fig)

                    # Ensure all are 1D arrays/lists and same length for predictions table
                    dates = pd.to_datetime(df_last['Date'][1:]).dt.strftime('%Y-%m-%d').to_list()
                    actual = df_last['Close'][1:].to_list()
                    predicted = y_pred.flatten().tolist() if hasattr(y_pred, 'flatten') else list(y_pred)

                    # Only display the predictions table if lengths match, otherwise do nothing (no warning)
                    if len(dates) == len(actual) == len(predicted):
                        pred_df = pd.DataFrame({
                            'Date': dates,
                            'Actual Close': actual,
                            'Predicted Close': predicted
                        })
                        st.subheader("Predictions Table (Last 30 Days)")
                        st.dataframe(pred_df)
                    # else: do nothing

                    # Show RMSE
                    rmse = mean_squared_error(df_last['Close'][1:], y_pred, squared=False)
                    st.info(f"Model RMSE: {rmse:.2f}")
        except Exception as e:
            st.error(f"Error fetching or processing data: {e}")
else:
    st.info("⬅️ Use the sidebar to upload a dataset or run a stock prediction!")

# --- Footer ---
st.markdown("""
---
<div style='text-align:center; color: #00F2FE;'>
    Made with <span style='color:#43e97b;'>❤️</span> by Mashie | Powered by Streamlit, yfinance, and Plotly
</div>
""", unsafe_allow_html=True) 