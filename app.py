# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Interactive DCF & Company Valuation", layout="wide")
st.title("Equity Valuation & DCF Modeling Tool")

# --- User Inputs ---
st.sidebar.header("Company & Forecast Inputs")
ticker = st.sidebar.text_input("Company Ticker", value="AAPL").upper()
company = yf.Ticker(ticker)

# Forecast Inputs
years = st.sidebar.slider("Forecast Years", 3, 10, 5)
default_wacc = 0.08
wacc = st.sidebar.slider("Discount Rate / WACC (%)", 0.0, 30.0, float(default_wacc*100))/100
terminal_growth = st.sidebar.slider("Terminal Growth Rate (%)", -2.0, 6.0, 2.5)/100
fcf_growth_min = st.sidebar.slider("FCF Growth Min (%)", -10, 30, 5)/100
fcf_growth_max = st.sidebar.slider("FCF Growth Max (%)", -10, 30, 10)/100

# --- Fetch Financial Data ---
st.subheader(f"{ticker} Financial Data")
with st.spinner("Fetching financial statements..."):
    try:
        income = company.financials.T
        balance = company.balance_sheet.T
        cashflow = company.cashflow.T
        info = company.info
        hist = company.history(period="5y")
    except:
        st.error("Could not fetch financial data. Check ticker symbol.")
        st.stop()

st.markdown("### Income Statement")
st.dataframe(income.style.format("{:,.0f}"), use_container_width=True)
st.markdown("### Balance Sheet")
st.dataframe(balance.style.format("{:,.0f}"), use_container_width=True)
st.markdown("### Cash Flow Statement")
st.dataframe(cashflow.style.format("{:,.0f}"), use_container_width=True)
st.markdown("### Historical Stock Price")
st.line_chart(hist["Close"])

# --- FCF Calculation ---
st.subheader("Free Cash Flow Forecast")
if 'TotalCashFromOperatingActivities' in cashflow.columns and 'CapitalExpenditures' in cashflow.columns:
    last_fcf = cashflow['TotalCashFromOperatingActivities'][-1] - abs(cashflow['CapitalExpenditures'][-1])
else:
    last_fcf = cashflow.iloc[-1].get('TotalCashFromOperatingActivities',0)

fcf_forecast = []
prev_fcf = last_fcf
for i in range(years):
    growth = np.random.uniform(fcf_growth_min, fcf_growth_max)
    prev_fcf = prev_fcf*(1 + growth)
    fcf_forecast.append(prev_fcf)

dfc_df = pd.DataFrame({"Year": range(1, years+1), "Forecasted FCF": fcf_forecast})
st.dataframe(dfc_df.style.format("{:,.0f}"), use_container_width=True)

# --- DCF Valuation ---
discounted_fcf = [fcf / (1 + wacc)**(i+1) for i, fcf in enumerate(fcf_forecast)]
terminal_value = fcf_forecast[-1]*(1+terminal_growth)/(wacc - terminal_growth)
terminal_value_discounted = terminal_value / (1 + wacc)**years
dcf_value = sum(discounted_fcf) + terminal_value_discounted

st.subheader("DCF Results")
st.metric("Enterprise Value (DCF)", f"${dcf_value:,.0f}")
st.write("Terminal Value (Discounted):", f"${terminal_value_discounted:,.0f}")

# --- Interactive Plot ---
st.subheader("DCF Visualization")
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"Year {i+1}" for i in range(years)], y=discounted_fcf, name="Discounted FCF"))
fig.add_trace(go.Bar(x=["Terminal"], y=[terminal_value_discounted], name="Terminal Value"))
fig.update_layout(title=f"{ticker} DCF Components", barmode='stack', yaxis_title="Value ($)")
st.plotly_chart(fig, use_container_width=True)

# --- Comparable Companies ---
st.subheader("Comparable Companies Analysis")
manual_comps = st.text_input("Enter comparable tickers (comma-separated)", value="MSFT,GOOGL,AMZN")
manual_comps_list = [x.strip().upper() for x in manual_comps.split(",")]

comp_data = []
for comp in manual_comps_list:
    try:
        c = yf.Ticker(comp)
        price = c.history(period="1d")['Close'][-1]
        market_cap = c.info.get('marketCap', np.nan)
        pe_ratio = c.info.get('trailingPE', np.nan)
        pb_ratio = c.info.get('priceToBook', np.nan)
        comp_data.append({"Ticker": comp, "Price": price, "Market Cap": market_cap, "P/E": pe_ratio, "P/B": pb_ratio})
    except:
        comp_data.append({"Ticker": comp, "Price": np.nan, "Market Cap": np.nan, "P/E": np.nan, "P/B": np.nan})

st.dataframe(pd.DataFrame(comp_data).style.format({"Price":"${:,.2f}", "Market Cap":"${:,.0f}", "P/E":"{:.2f}", "P/B":"{:.2f}"}), use_container_width=True)

# --- Optional: Options Greeks ---
st.subheader("Options Greeks (European Call Example)")
S = st.number_input("Stock Price", value=float(hist["Close"][-1]))
K = st.number_input("Strike Price", value=S)
T = st.number_input("Time to Expiration (years)", value=0.5)
sigma = st.number_input("Volatility (%)", value=0.3)/100
r = wacc

d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
delta = norm.cdf(d1)
gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
vega = S*norm.pdf(d1)*np.sqrt(T)
theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
rho = K*T*np.exp(-r*T)*norm.cdf(d2)

st.write(f"Call Price: ${call_price:,.2f}")
st.write(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}, Theta: {theta:.4f}, Rho: {rho:.4f}")

