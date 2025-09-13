# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Equity Valuation & DCF", layout="wide")

st.title("Interactive Company Valuation Dashboard")

# --- User Inputs ---
ticker = st.text_input("Enter the company ticker", value="AAPL").upper()
company = yf.Ticker(ticker)

st.subheader("Discount & Terminal Growth")
default_wacc = 0.08  # Default WACC
discount = st.slider("Discount rate / WACC (%)", 0.0, 30.0, float(default_wacc*100))/100
terminal_growth = st.slider("Terminal Growth Rate (%)", -2.0, 6.0, 2.5)/100

st.subheader("Forecast Period")
years = st.slider("Number of forecast years", 3, 10, 5)

# --- Pull Financials ---
with st.spinner("Fetching financial data..."):
    try:
        income = company.financials.T
        balance = company.balance_sheet.T
        cashflow = company.cashflow.T
    except:
        st.error("Could not fetch financial data. Check ticker symbol.")
        st.stop()

# --- Display Financials ---
st.subheader("Income Statement")
st.dataframe(income.style.format("{:,.0f}"), use_container_width=True)

st.subheader("Balance Sheet")
st.dataframe(balance.style.format("{:,.0f}"), use_container_width=True)

st.subheader("Cash Flow Statement")
st.dataframe(cashflow.style.format("{:,.0f}"), use_container_width=True)

# --- Historical Data ---
st.subheader("Historical Stock Data")
hist = company.history(period="5y")
st.line_chart(hist["Close"])

# --- DCF Forecast ---
st.subheader("DCF Forecast")
last_fcf = cashflow['FreeCashFlow'][-1] if 'FreeCashFlow' in cashflow.columns else cashflow.iloc[-1].get('TotalCashFromOperatingActivities',0)
growth_rates = st.slider("Expected FCF growth each year (%)", -10, 30, (5,10))
growth_rates = [r/100 for r in growth_rates]

fcf_forecast = []
prev = last_fcf
for i in range(years):
    prev = prev*(1+np.mean(growth_rates))
    fcf_forecast.append(prev)

dfc_df = pd.DataFrame({
    "Year": range(1, years+1),
    "Forecasted FCF": fcf_forecast
})

st.dataframe(dfc_df.style.format("{:,.0f}"), use_container_width=True)

# --- DCF Valuation ---
discounted_fcf = [fcf / (1 + discount)**(i+1) for i, fcf in enumerate(fcf_forecast)]
terminal_value = fcf_forecast[-1]*(1+terminal_growth)/ (discount - terminal_growth)
terminal_value_discounted = terminal_value / (1 + discount)**years
dcf_value = sum(discounted_fcf) + terminal_value_discounted

st.metric("DCF Equity Value", f"${dcf_value:,.0f}")

# --- Comparables ---
st.subheader("Comparable Companies")
try:
    comps = company.recommendations
except:
    comps = pd.DataFrame()

if comps.empty:
    st.info("No comparables available via Yahoo Finance. You can manually input tickers below.")
else:
    st.dataframe(comps, use_container_width=True)

manual_comps = st.text_input("Enter additional comparable tickers (comma-separated)", value="MSFT,GOOGL,AMZN")
manual_comps_list = [x.strip().upper() for x in manual_comps.split(",")]

st.write("Manual comparables:", manual_comps_list)

# --- Interactive Plot ---
st.subheader("DCF Visualization")
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"Year {i+1}" for i in range(years)], y=discounted_fcf, name="Discounted FCF"))
fig.add_trace(go.Bar(x=["Terminal"], y=[terminal_value_discounted], name="Terminal Value"))
fig.update_layout(barmode='stack', title="DCF Components", yaxis_title="Value ($)")
st.plotly_chart(fig, use_container_width=True)

# --- Options Greeks Example (Optional) ---
st.subheader("Options Greeks (European Call)")
S = st.number_input("Current Stock Price", value=float(hist["Close"][-1]))
K = st.number_input("Strike Price", value=S)
T = st.number_input("Time to Expiration (years)", value=0.5)
r = discount
sigma = st.number_input("Volatility (%)", value=0.3)/100

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
