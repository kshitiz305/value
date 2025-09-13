# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import requests

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Analyst Terminal", layout="wide")
st.title("💹 Analyst Terminal — Equity Valuation & Options")
st.caption('By [Navjot Dhah](https://www.linkedin.com/in/navjot-dhah-57870b238)')

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
projection_years = st.sidebar.selectbox("Projection Years", [3,5,7,10], index=1)

# -------------------------
# Fetch YFinance Data
# -------------------------
@st.cache_data(ttl=300)
def fetch_yf(t):
    tk = yf.Ticker(t)
    info = tk.info if hasattr(tk,"info") else {}
    fin = tk.financials if hasattr(tk,"financials") else pd.DataFrame()
    bs = tk.balance_sheet if hasattr(tk,"balance_sheet") else pd.DataFrame()
    cf = tk.cashflow if hasattr(tk,"cashflow") else pd.DataFrame()
    hist = tk.history(period="5y") if hasattr(tk,"history") else pd.DataFrame()
    return info, fin, bs, cf, hist

info, fin, bs, cf, hist = fetch_yf(ticker)

# -------------------------
# Company Overview
# -------------------------
st.header(f"{info.get('shortName', ticker)} — {ticker}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"${info.get('currentPrice','N/A')}")
col2.metric("Market Cap", f"${info.get('marketCap','N/A')}")
col3.metric("Shares Outstanding", f"{info.get('sharesOutstanding','N/A')}")
col4.metric("Beta", info.get('beta','N/A'))

# Extra info
st.write("**Sector:**", info.get("sector","N/A"), " | **Industry:**", info.get("industry","N/A"))
st.write("**P/E Ratio:**", info.get("trailingPE","N/A"), " | **Forward P/E:**", info.get("forwardPE","N/A"))
st.write("**Dividend Yield:**", info.get("dividendYield","N/A"))

# -------------------------
# Historical Price Chart
# -------------------------
st.subheader("Price Chart (5 Years)")
if not hist.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical price data available.")

# -------------------------
# Financial Statements
# -------------------------
st.subheader("Financial Statements")
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Income Statement**")
    if not fin.empty:
        st.dataframe(fin.T, use_container_width=True)
    else:
        st.info("N/A")
with f2:
    st.markdown("**Balance Sheet**")
    if not bs.empty:
        st.dataframe(bs.T, use_container_width=True)
    else:
        st.info("N/A")
with f3:
    st.markdown("**Cash Flow**")
    if not cf.empty:
        st.dataframe(cf.T, use_container_width=True)
    else:
        st.info("N/A")

# -------------------------
# 3-Statement Integration + FCF Projection
# -------------------------
st.subheader("FCF Projection (Analyst Inputs)")
last_revenue = st.number_input("Most recent Revenue", value=info.get("totalRevenue", 1_000_000_000))
rev_growth = st.slider("Revenue Growth (%)", -10, 30, 5)/100
ebit_margin = st.slider("EBIT Margin (%)", -50, 50, 15)/100
tax_rate = st.slider("Tax Rate (%)", 0, 50, 25)/100
depreciation = st.number_input("Depreciation (latest)", value=500_000_000)
capex = st.number_input("CAPEX (latest)", value=300_000_000)
change_wc = st.number_input("Change in Working Capital (latest)", value=50_000_000)

years = projection_years
revenue_proj = [last_revenue*(1+rev_growth)**i for i in range(1, years+1)]
ebit_proj = [r*ebit_margin for r in revenue_proj]
tax_proj = [e*tax_rate for e in ebit_proj]
fcf_proj = [ebit_proj[i]-tax_proj[i]+depreciation-capex-change_wc for i in range(years)]

# Discount inputs
discount_rate = st.slider("Discount Rate / WACC (%)", 0.1, 30.0, 9.0)/100
terminal_growth = st.slider("Terminal Growth Rate (%)", -2, 6, 2.5)/100
terminal_value = fcf_proj[-1]*(1+terminal_growth)/(discount_rate-terminal_growth)
pv_fcf = [fcf_proj[i]/((1+discount_rate)**(i+1)) for i in range(years)]
pv_terminal = terminal_value/((1+discount_rate)**years)
enterprise_value = sum(pv_fcf)+pv_terminal

st.metric("Enterprise Value (DCF)", f"${enterprise_value:,.0f}")

# DCF chart
fig_dcf = go.Figure()
fig_dcf.add_trace(go.Bar(x=[f"Y{i}" for i in range(1, years+1)], y=pv_fcf, name="Discounted FCF"))
fig_dcf.add_trace(go.Bar(x=["Terminal"], y=[pv_terminal], name="Terminal Value"))
fig_dcf.update_layout(barmode="stack", title="DCF Projection")
st.plotly_chart(fig_dcf, use_container_width=True)

# Sensitivity Table (Discount vs Terminal Growth)
dg = np.arange(discount_rate-0.02, discount_rate+0.02, 0.01)
tg = np.arange(terminal_growth-0.01, terminal_growth+0.01, 0.005)
sens_table = pd.DataFrame(index=[f"{d*100:.1f}%" for d in dg], columns=[f"{t*100:.1f}%" for t in tg])
for i, d in enumerate(dg):
    for j, t in enumerate(tg):
        tv = fcf_proj[-1]*(1+t)/(d-t)
        pv_tv = tv/((1+d)**years)
        pv_fcfs = sum([fcf_proj[k]/((1+d)**(k+1)) for k in range(years)])
        sens_table.iloc[i,j] = pv_fcfs+pv_tv
st.subheader("DCF Sensitivity Table (Enterprise Value)")
st.dataframe(sens_table)

# -------------------------
# Options (Black-Scholes)
# -------------------------
st.subheader("Options Pricing (Black–Scholes)")
S = st.number_input("Underlying Price (S)", value=float(info.get('currentPrice', 100)))
K = st.number_input("Strike (K)", value=float(info.get('currentPrice', 100)))
days = st.number_input("Days to Expiry", 1, 3650, 30)
r = st.number_input("Risk-free rate (%)", 0.0, 10.0, 0.5)/100
sigma = st.number_input("Volatility (%)", 0.0, 200.0, 25.0)/100
T = days/365.0
d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
delta_call = norm.cdf(d1)
delta_put = delta_call-1
gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
vega = S*norm.pdf(d1)*np.sqrt(T)
theta_call = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
theta_put = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
st.write(f"Call ≈ ${call:,.2f}, Put ≈ ${put:,.2f}")
st.write(f"Δ Call={delta_call:.3f}, Δ Put={delta_put:.3f}, Γ={gamma:.3f}, Vega={vega:.2f}, θ Call={theta_call:.2f}, θ Put={theta_put:.2f}")

# -------------------------
# Comparables
# -------------------------
st.subheader("Comparable Companies")
default_comps = ["MSFT","GOOGL","AMZN"]
comps = st.multiselect("Select peers", options=default_comps, default=default_comps)
for comp in comps:
    cinfo, _, _, _, _ = fetch_yf(comp)
    st.write(f"{comp}: Price=${cinfo.get('currentPrice','N/A')}, Market Cap=${cinfo.get('marketCap','N/A')}, P/E={cinfo.get('trailingPE','N/A')}")

# -------------------------
# News Feed
# -------------------------
st.subheader("Latest News")
try:
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
    resp = requests.get(url, timeout=6).json()
    items = resp.get("news", [])[:5]
    for n in items:
        st.markdown(f"- [{n.get('title','No title')}]({n.get('link','#')})")
except:
    st.info("No news available.")

