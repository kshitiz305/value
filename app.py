import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="Equity Analysis Tool", layout="wide")

demo = 'a50f972afe6637de4b75c22b25793300'

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose Analysis Type", ["LBO", "DCF", "Comparables"])

company = st.sidebar.text_input("Enter Company Ticker", "GOOG")

# -------------------------------
# Cached API Functions
# -------------------------------
@st.cache_data
def get_company_profile(ticker):
    url = f'https://financialmodelingprep.com/api/v3/company/profile/{ticker}?apikey={demo}'
    return requests.get(url).json()

@st.cache_data
def get_income_statement(ticker):
    url = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={demo}'
    return requests.get(url).json()

@st.cache_data
def get_balance_sheet(ticker):
    url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={demo}'
    return requests.get(url).json()

@st.cache_data
def get_enterprise_values(ticker):
    url = f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?apikey={demo}'
    return requests.get(url).json()

# -------------------------------
# Fetch Data
# -------------------------------
profile = get_company_profile(company)
IS = get_income_statement(company)
BS = get_balance_sheet(company)
EV = get_enterprise_values(company)

# -------------------------------
# LBO Section
# -------------------------------
if option == "LBO":
    st.header("Leveraged Buyout (LBO) Analysis")

    st.subheader("Inputs")
    purchase_price = st.number_input("Purchase Price ($M)", value=1000.0)
    debt_equity_ratio = st.slider("Debt/Equity Ratio", 0.0, 1.0, 0.6)
    interest_rate = st.number_input("Interest Rate on Debt (%)", value=6.0)/100
    exit_multiple = st.number_input("Exit EBITDA Multiple", value=8.0)
    forecast_years = st.slider("Forecast Years", 3, 10, 5)

    # Optionally pull EBITDA from API
    EBITDA = IS[0]['ebitda'] if 'ebitda' in IS[0] else 100

    st.write(f"Company EBITDA: ${EBITDA:,.0f}M")

    # Debt / Equity split
    debt = purchase_price * debt_equity_ratio
    equity = purchase_price - debt

    st.write(f"Debt: ${debt:,.0f}M | Equity: ${equity:,.0f}M")

    # Forecast cash flows
    growth_rate = st.number_input("EBITDA Growth Rate (%)", value=5.0)/100
    CF = [EBITDA * (1+growth_rate)**i for i in range(1, forecast_years+1)]
    debt_balance = debt
    for i, cf in enumerate(CF):
        debt_payment = min(cf*0.8, debt_balance)  # simple debt paydown assumption
        debt_balance -= debt_payment
        CF[i] -= debt_payment

    exit_value = CF[-1] * exit_multiple / EBITDA * EBITDA
    equity_exit = exit_value - debt_balance

    st.subheader("Outputs")
    st.write(f"Equity Value at Exit: ${equity_exit:,.0f}M")
    st.write(f"Estimated IRR (Simplified): {((equity_exit / equity)**(1/forecast_years)-1)*100:.2f}%")

# -------------------------------
# DCF Section
# -------------------------------
elif option == "DCF":
    st.header("Discounted Cash Flow (DCF) Analysis")

    st.subheader("Inputs")
    revenue = IS[0]['revenue'] if 'revenue' in IS[0] else 1000
    rev_growth = st.number_input("Revenue Growth Rate (%)", value=5.0)/100
    wacc = st.number_input("WACC (%)", value=8.0)/100
    terminal_growth = st.number_input("Terminal Growth (%)", value=3.0)/100
    forecast_years = st.slider("Forecast Years", 3, 10, 5)

    # Forecast Free Cash Flows
    FCF = [revenue*(1+rev_growth)**i * 0.15 for i in range(1, forecast_years+1)]  # simple 15% FCF margin
    PV_FCF = [fcf/(1+wacc)**i for i, fcf in enumerate(FCF, start=1)]
    terminal_value = FCF[-1]*(1+terminal_growth)/(wacc-terminal_growth)
    PV_terminal = terminal_value/(1+wacc)**forecast_years
    enterprise_value = sum(PV_FCF) + PV_terminal
    st.write(f"Estimated Enterprise Value: ${enterprise_value:,.0f}M")

# -------------------------------
# Comparables Section
# -------------------------------
elif option == "Comparables":
    st.header("Comparable Company Analysis")

    st.subheader("Inputs")
    competitor_list = st.text_area("Enter Competitor Tickers (comma separated) or leave blank for auto-fetch")
    competitors = [c.strip() for c in competitor_list.split(",") if c.strip()]

    if not competitors:
        # TODO: auto-fetch competitors from FMP (placeholder)
        competitors = ["AAPL", "MSFT", "AMZN"]  

    comp_data = []
    for comp in competitors:
        comp_profile = get_company_profile(comp)
        market_cap = comp_profile['profile'].get('mktCap', 0)
        comp_data.append({'Ticker': comp, 'Market Cap': market_cap})

    df_comp = pd.DataFrame(comp_data)
    st.write("Comparable Companies:")
    st.dataframe(df_comp)
