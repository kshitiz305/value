import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# -------------------------- #
#           CONFIG           #
# -------------------------- #
st.set_page_config(page_title="Valuation & DCF Dashboard", layout="wide")

# -------------------------- #
#       UTILS / CACHING      #
# -------------------------- #
@st.cache_resource(show_spinner=False)
def fetch_all(ticker: str):
    """
    Use cache_resource to avoid pickling issues.
    Return only plain-Python serializable items (dicts/DataFrames) and coerce fast_info to dict.
    """
    t = yf.Ticker(ticker)

    # info can be a normal dict or cause network exceptions; guard it
    try:
        info = t.info if hasattr(t, "info") else {}
        if not isinstance(info, dict):
            # Some yfinance versions may return a Mapping-like; best effort to coerce
            info = dict(info)
    except Exception:
        info = {}

    # fast_info is often a custom object; force it to a dict
    try:
        fi = getattr(t, "fast_info", {}) or {}
        if isinstance(fi, dict):
            fast = fi
        elif hasattr(fi, "items"):
            fast = dict(fi)
        elif hasattr(fi, "__dict__"):
            fast = dict(fi.__dict__)
        else:
            fast = {}
    except Exception:
        fast = {}

    # History (DataFrame)
    try:
        hist = t.history(period="10y", auto_adjust=False)
    except Exception:
        hist = pd.DataFrame()

    # Statements (DataFrames)
    fin_a = t.financials if hasattr(t, "financials") and isinstance(t.financials, pd.DataFrame) else pd.DataFrame()
    fin_q = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
    bs_a  = t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame()
    bs_q  = t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame()
    cf_a  = t.cashflow if hasattr(t, "cashflow") else pd.DataFrame()
    cf_q  = t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame()

    return info, fast, hist, fin_a, fin_q, bs_a, bs_q, cf_a, cf_q

def _safe_get(df: pd.DataFrame, row_name: str):
    """Return series (across periods) for a row_name if present, else empty Series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    candidates = [row_name,
                  row_name.lower(),
                  row_name.replace(" ", "_"),
                  row_name.title(),
                  row_name.upper()]
    for key in candidates:
        if key in df.index:
            return df.loc[key]
    for idx in df.index:
        if row_name.lower() in str(idx).lower():
            return df.loc[idx]
    return pd.Series(dtype=float)

def series_to_annual_df(s: pd.Series, label: str):
    if s.empty:
        return pd.DataFrame()
    out = pd.DataFrame({label: s})
    out.index.name = "Period"
    return out

def trailing_twelve_months(q_series: pd.Series, min_quarters=4):
    """Sum of the last 4 quarters if available."""
    if q_series is None or q_series.empty:
        return np.nan
    q_series = q_series.sort_index()
    if len(q_series) < min_quarters:
        return np.nan
    return float(q_series.iloc[-4:].sum())

def millions(x):  # helper for formatting
    try:
        return x / 1_000_000.0
    except Exception:
        return np.nan

# -------------------------- #
#           SIDEBAR          #
# -------------------------- #
st.sidebar.title("🔎 Stock & Settings")
ticker = st.sidebar.text_input("Primary Ticker (e.g., AAPL, MSFT, TSLA):", value="AAPL").upper().strip()
peer_input = st.sidebar.text_input("Peer Tickers (comma-separated):", value="MSFT, GOOGL, AMZN")
peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("DCF Assumptions (Equity DCF / FCFE)")
rf = st.sidebar.number_input("Risk-free rate (%)", value=4.0, step=0.25)
beta_override = st.sidebar.text_input("Beta (blank = use Yahoo beta):", value="")
mkt_prem = st.sidebar.number_input("Market risk premium (%)", value=5.0, step=0.25)
cost_equity_override = st.sidebar.text_input("Override Cost of Equity (%) [optional]:", value="")
start_fcfe_growth = st.sidebar.number_input("Years 1–5 FCFE growth (%)", value=8.0, step=0.5)
terminal_growth = st.sidebar.number_input("Terminal growth (%)", value=2.5, step=0.25)
years_forecast = st.sidebar.slider("Forecast horizon (years)", 3, 10, 5)
shares_override = st.sidebar.text_input("Shares Outstanding (override, in shares) [optional]:", value="")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Provide realistic peers for a stronger comparables section.")

# -------------------------- #
#         MAIN HEADER        #
# -------------------------- #
st.title("📊 Company Analyzer & Valuation (Streamlit)")
st.write(
    "Enter a ticker on the left, then explore financial statements, business overview, ratios, a detailed DCF, "
    "and comparables vs peers. Data source: Yahoo Finance via `yfinance`."
)

# -------------------------- #
#       FETCH PRIMARY        #
# -------------------------- #
if not ticker:
    st.stop()

info, fast, hist, fin_a, fin_q, bs_a, bs_q, cf_a, cf_q = fetch_all(ticker)

company_name = info.get("longName") or info.get("shortName") or ticker
sector = info.get("sector", "—")
industry = info.get("industry", "—")
country = info.get("country", "—")
currency = info.get("currency", "USD")

colA, colB, colC, colD = st.columns(4)
colA.metric("Company", company_name)
colB.metric("Sector", sector)
colC.metric("Industry", industry)
colD.metric("Reporting Currency", currency)

# Price chart
if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
    price_fig = px.line(hist.reset_index(), x="Date", y="Close", title=f"{ticker} Price (10y)")
    st.plotly_chart(price_fig, use_container_width=True)
else:
    st.info("No price history available.")

# -------------------------- #
#     FINANCIAL STATEMENTS   #
# -------------------------- #
st.header("📒 Financial Statements (Annual & Quarterly)")

def show_statement(name, annual_df, quarterly_df):
    st.subheader(name)
    c1, c2 = st.columns(2)
    with c1:
        if annual_df is not None and not annual_df.empty:
            st.caption("Annual")
            st.dataframe(annual_df.fillna("").style.format("{:,.0f}"), use_container_width=True, height=360)
        else:
            st.info("Annual data unavailable.")
    with c2:
        if quarterly_df is not None and not quarterly_df.empty:
            st.caption("Quarterly")
            st.dataframe(quarterly_df.fillna("").style.format("{:,.0f}"), use_container_width=True, height=360)
        else:
            st.info("Quarterly data unavailable.")

show_statement("Income Statement", fin_a, fin_q)
show_statement("Balance Sheet", bs_a, bs_q)
show_statement("Cash Flow Statement", cf_a, cf_q)

# -------------------------- #
#     BASIC DERIVED METRICS  #
# -------------------------- #
st.header("🧮 Key Metrics & Trends")

rev_a = _safe_get(fin_a, "Total Revenue")
rev_q = _safe_get(fin_q, "Total Revenue")
ebitda_a = _safe_get(fin_a, "Ebitda")
ebitda_q = _safe_get(fin_q, "Ebitda")
net_income_a = _safe_get(fin_a, "Net Income")
net_income_q = _safe_get(fin_q, "Net Income")
gross_profit_a = _safe_get(fin_a, "Gross Profit")
oper_income_a = _safe_get(fin_a, "Operating Income")

# TTM values (quarterly sums)
rev_ttm = trailing_twelve_months(rev_q)
ebitda_ttm = trailing_twelve_months(ebitda_q)
net_inc_ttm = trailing_twelve_months(net_income_q)

# Margins (annual, most recent)
def last_non_nan(s: pd.Series):
    if s is None or s.empty:
        return np.nan
    s = s.dropna()
    return s.iloc[-1] if not s.empty else np.nan

latest_rev = last_non_nan(rev_a)
latest_gross = last_non_nan(gross_profit_a)
latest_oper = last_non_nan(oper_income_a)
latest_ebitda = last_non_nan(ebitda_a)
latest_net = last_non_nan(net_income_a)

gross_margin = (latest_gross / latest_rev) if latest_gross and latest_rev else np.nan
oper_margin  = (latest_oper / latest_rev) if latest_oper and latest_rev else np.nan
ebitda_margin = (latest_ebitda / latest_rev) if latest_ebitda and latest_rev else np.nan
net_margin   = (latest_net / latest_rev) if latest_net and latest_rev else np.nan

# Balance sheet snapshot (latest)
total_debt = 0.0
cash = 0.0
shares = info.get("sharesOutstanding") or fast.get("shares_outstanding")
if shares_override.strip():
    try:
        shares = float(shares_override.replace(",", ""))
    except Exception:
        pass

if not bs_a.empty:
    debt_candidates = ["Total Debt", "Short Long Term Debt", "Short Long-Term Debt", "Long Term Debt", "Long-Term Debt", "Current Debt"]
    for d in debt_candidates:
        s = _safe_get(bs_a, d)
        if not s.empty:
            total_debt += float(last_non_nan(s) or 0.0)
    cash_candidates = ["Cash", "Cash And Cash Equivalents", "Cash And Short Term Investments"]
    for c in cash_candidates:
        s = _safe_get(bs_a, c)
        if not s.empty:
            cash = float(last_non_nan(s))
            break

market_cap = info.get("marketCap") or fast.get("market_cap")
price = fast.get("last_price") or info.get("currentPrice")

# KPIs row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Revenue (TTM, $M)", f"{millions(rev_ttm):,.1f}" if rev_ttm==rev_ttm else "—")
k2.metric("EBITDA (TTM, $M)", f"{millions(ebitda_ttm):,.1f}" if ebitda_ttm==ebitda_ttm else "—")
k3.metric("Net Income (TTM, $M)", f"{millions(net_inc_ttm):,.1f}" if net_inc_ttm==net_inc_ttm else "—")
k4.metric("Gross Margin", f"{100*gross_margin:,.1f}%" if gross_margin==gross_margin else "—")
k5.metric("Operating Margin", f"{100*oper_margin:,.1f}%" if oper_margin==oper_margin else "—")
k6.metric("Net Margin", f"{100*net_margin:,.1f}%" if net_margin==net_margin else "—")

# Trend charts (Annual)
def plot_trend(series: pd.Series, title: str, ytitle: str):
    if series is None or series.empty:
        st.info(f"{title}: no data.")
        return
    df = series_to_annual_df(series, ytitle)
    df = df.reset_index().rename(columns={"index": "Period"})
    try:
        df["Period"] = pd.to_datetime(df["Period"])
    except Exception:
        pass
    fig = px.bar(df, x="Period", y=ytitle, title=title)
    st.plotly_chart(fig, use_container_width=True)

cA, cB, cC = st.columns(3)
with cA:
    plot_trend(rev_a, "Revenue (Annual)", "Revenue")
with cB:
    plot_trend(ebitda_a, "EBITDA (Annual)", "EBITDA")
with cC:
    plot_trend(net_income_a, "Net Income (Annual)", "Net Income")

# -------------------------- #
#            DCF             #
# -------------------------- #
st.header("💰 Equity DCF (FCFE)")

# FCFE proxy = CFO - CapEx (TTM)
cfo_q = _safe_get(cf_q, "Total Cash From Operating Activities")
capex_q = _safe_get(cf_q, "Capital Expenditures")
fcfe_ttm = np.nan
if not cfo_q.empty and not capex_q.empty:
    cfo_ttm = trailing_twelve_months(cfo_q)
    capex_ttm = trailing_twelve_months(capex_q)
    if cfo_ttm==cfo_ttm and capex_ttm==capex_ttm:
        fcfe_ttm = cfo_ttm + capex_ttm  # capex is negative in Yahoo
else:
    # fallback: use annual last year
    cfo_a = _safe_get(cf_a, "Total Cash From Operating Activities")
    capex_a = _safe_get(cf_a, "Capital Expenditures")
    if not cfo_a.empty and not capex_a.empty:
        fcfe_ttm = float(last_non_nan(cfo_a) + last_non_nan(capex_a))

if fcfe_ttm != fcfe_ttm:
    st.warning("Could not compute FCFE (CFO - CapEx). DCF may be limited.")
else:
    st.caption(f"FCFE (TTM): {fcfe_ttm:,.0f} {currency}")

# Cost of Equity via CAPM
beta = None
try:
    beta = info.get("beta")
except Exception:
    beta = None
if beta_override.strip():
    try:
        beta = float(beta_override)
    except Exception:
        pass

capm_cost = None
if beta is not None:
    capm_cost = rf/100.0 + beta*(mkt_prem/100.0)  # decimal

if cost_equity_override.strip():
    try:
        cost_equity = float(cost_equity_override)/100.0
    except Exception:
        cost_equity = capm_cost or 0.10
else:
    cost_equity = capm_cost or 0.10

st.caption(
    f"Cost of Equity used: {100*cost_equity:,.2f}% "
    + (f"(CAPM with β={beta:.2f})" if capm_cost and not cost_equity_override else "(override)")
)

def project_fcfe(base_fcfe, growth_pct, years):
    fcfe_list = []
    fcfe = float(base_fcfe)
    g = growth_pct/100.0
    for y in range(1, years+1):
        fcfe = fcfe * (1.0 + g)
        fcfe_list.append(fcfe)
    return fcfe_list

intrinsic_ps = None
if fcfe_ttm==fcfe_ttm and shares:
    proj = project_fcfe(fcfe_ttm, start_fcfe_growth, years_forecast)
    # Terminal value (Gordon on Year N FCFE)
    g_t = terminal_growth/100.0
    ke = max(cost_equity, g_t + 0.0001)  # avoid div by zero/negative spread
    tv = proj[-1] * (1.0 + g_t) / (ke - g_t)

    # Discount factors
    years = np.arange(1, years_forecast+1)
    disc = 1.0 / (1.0 + ke)**years
    pv_cf = (np.array(proj) * disc).sum()
    pv_tv = tv / (1.0 + ke)**years_forecast
    equity_value = pv_cf + pv_tv
    intrinsic_ps = equity_value / float(shares)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PV of Forecast FCFE", f"{pv_cf:,.0f} {currency}")
    c2.metric("PV of Terminal Value", f"{pv_tv:,.0f} {currency}")
    c3.metric("Equity Value", f"{equity_value:,.0f} {currency}")
    c4.metric("Intrinsic Value / Share", f"{intrinsic_ps:,.2f} {currency}")

    # Plots
    df_proj = pd.DataFrame({"Year": years, "FCFE": proj, "Discount Factor": disc, "PV": np.array(proj)*disc})
    fcfe_fig = px.bar(df_proj, x="Year", y="FCFE", title="Projected FCFE")
    pv_fig = px.bar(df_proj, x="Year", y="PV", title="Present Value of FCFE")
    st.plotly_chart(fcfe_fig, use_container_width=True)
    st.plotly_chart(pv_fig, use_container_width=True)

    # Sensitivity (heatmap) for Terminal g vs Cost of Equity
    tg_range = np.linspace(max(g_t-0.02, 0.0), g_t+0.02, 5)
    ke_range = np.linspace(max(ke-0.03, 0.06), min(ke+0.03, 0.20), 5)
    z = []
    for ke_i in ke_range:
        row = []
        for tg_i in tg_range:
            ke_i = max(ke_i, tg_i+0.0001)
            tv_i = proj[-1]*(1+tg_i)/(ke_i - tg_i)
            pv_tv_i = tv_i / (1.0 + ke_i)**years_forecast
            pv_cf_i = (np.array(proj) / (1.0 + ke_i)**years).sum()
            eq_i = pv_cf_i + pv_tv_i
            row.append(eq_i/float(shares))
        z.append(row)
    heat = go.Figure(data=go.Heatmap(
        z=z, x=[f"{x*100:.2f}%" for x in tg_range], y=[f"{y*100:.2f}%" for y in ke_range],
        colorbar_title="Intrinsic / Share"
    ))
    heat.update_layout(title="Sensitivity: Intrinsic / Share vs Terminal g and Cost of Equity",
                       xaxis_title="Terminal Growth", yaxis_title="Cost of Equity")
    st.plotly_chart(heat, use_container_width=True)
else:
    st.info("Need FCFE and shares outstanding to compute DCF. Try providing overrides in the sidebar if missing.")

# -------------------------- #
#      BUSINESS OVERVIEW     #
# -------------------------- #
st.header("🏢 Business Overview (from Yahoo)")
desc = info.get("longBusinessSummary") or "No description available."
st.write(desc)

# -------------------------- #
#         COMPARABLES        #
# -------------------------- #
st.header("📈 Comparables (User-Provided Peers)")
def get_quick_snapshot(tix):
    t = yf.Ticker(tix)
    try:
        inf = t.info if hasattr(t, "info") else {}
        if not isinstance(inf, dict):
            inf = dict(inf)
    except Exception:
        inf = {}
    try:
        fi = getattr(t, "fast_info", {}) or {}
        if isinstance(fi, dict):
            fst = fi
        elif hasattr(fi, "items"):
            fst = dict(fi)
        elif hasattr(fi, "__dict__"):
            fst = dict(fi.__dict__)
        else:
            fst = {}
    except Exception:
        fst = {}

    fin_q = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
    fin_a = t.financials if hasattr(t, "financials") else pd.DataFrame()
    bs_a = t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame()

    # Try TTM metrics
    rev = trailing_twelve_months(_safe_get(fin_q, "Total Revenue"))
    ni  = trailing_twelve_months(_safe_get(fin_q, "Net Income"))
    ebitda = trailing_twelve_months(_safe_get(fin_q, "Ebitda"))
    if np.isnan(rev):
        rev = float(last_non_nan(_safe_get(fin_a, "Total Revenue")) or np.nan)
    if np.isnan(ni):
        ni = float(last_non_nan(_safe_get(fin_a, "Net Income")) or np.nan)
    if np.isnan(ebitda):
        ebitda = float(last_non_nan(_safe_get(fin_a, "Ebitda")) or np.nan)

    # Market data
    mcap = inf.get("marketCap") or fst.get("market_cap") or np.nan
    price = fst.get("last_price") or inf.get("currentPrice")
    shares_local = inf.get("sharesOutstanding") or fst.get("shares_outstanding")

    # Debt/Cash for EV
    total_debt = 0.0
    cash_local = 0.0
    if bs_a is not None and not bs_a.empty:
        for d in ["Total Debt", "Short Long Term Debt", "Short Long-Term Debt", "Long Term Debt", "Current Debt"]:
            s = _safe_get(bs_a, d)
            if not s.empty:
                total_debt += float(last_non_nan(s) or 0.0)
        for c in ["Cash", "Cash And Cash Equivalents", "Cash And Short Term Investments"]:
            s = _safe_get(bs_a, c)
            if not s.empty:
                cash_local = float(last_non_nan(s))
                break
    ev = (mcap or 0.0) + total_debt - cash_local

    # Multiples
    pe = (mcap / ni) if (mcap and ni and ni != 0) else np.nan
    ev_ebitda = (ev / ebitda) if (ev and ebitda and ebitda != 0) else np.nan
    ps = (mcap / rev) if (mcap and rev and rev != 0) else np.nan

    return {
        "Ticker": tix,
        "Price": price,
        "Market Cap": mcap,
        "EV": ev,
        "Revenue (TTM)": rev,
        "EBITDA (TTM)": ebitda,
        "Net Income (TTM)": ni,
        "Shares": shares_local,
        "P/E": pe,
        "EV/EBITDA": ev_ebitda,
        "P/S": ps
    }

if peers:
    rows = []
    for p in peers + [ticker]:
        try:
            rows.append(get_quick_snapshot(p))
        except Exception:
            pass
    if rows:
        comp_df = pd.DataFrame(rows).set_index("Ticker")
        st.dataframe(comp_df.style.format({
            "Price": "{:,.2f}",
            "Market Cap": "{:,.0f}",
            "EV": "{:,.0f}",
            "Revenue (TTM)": "{:,.0f}",
            "EBITDA (TTM)": "{:,.0f}",
            "Net Income (TTM)": "{:,.0f}",
            "P/E": "{:,.1f}",
            "EV/EBITDA": "{:,.1f}",
            "P/S": "{:,.1f}",
        }), use_container_width=True)

        # Visualize multiples distribution
        for metric in ["P/E", "EV/EBITDA", "P/S"]:
            clean = comp_df[metric].dropna()
            if not clean.empty:
                fig = px.box(clean.reset_index(), x="Ticker", y=metric, title=f"{metric} by Ticker")
                st.plotly_chart(fig, use_container_width=True)

        # Simple relative valuation (median peer multiple applied to company)
        st.subheader("Peer-Implied Value (simple)")
        try:
            peer_only = comp_df.drop(index=ticker, errors="ignore")
            med_pe = peer_only["P/E"].median(skipna=True)
            med_ev_ebitda = peer_only["EV/EBITDA"].median(skipna=True)
            med_ps = peer_only["P/S"].median(skipna=True)

            # Apply to the company
            base = comp_df.loc[ticker]
            implied_pe = (med_pe * base["Net Income (TTM)"]) if med_pe==med_pe else np.nan
            implied_ev = (med_ev_ebitda * base["EBITDA (TTM)"]) if med_ev_ebitda==med_ev_ebitda else np.nan
            implied_ps = (med_ps * base["Revenue (TTM)"]) if med_ps==med_ps else np.nan

            # Convert EV to equity (approx): EV - Net Debt
            net_debt = float(base["EV"] - base["Market Cap"]) if base["EV"]==base["EV"] and base["Market Cap"]==base["Market Cap"] else np.nan
            eq_from_ev = implied_ev - net_debt if implied_ev==implied_ev and net_debt==net_debt else np.nan

            # Per share (use the company's shares from comp table if present; else fall back to overall)
            shares_for_peer = base.get("Shares", np.nan)
            if isinstance(shares_for_peer, (int, float)) and shares_for_peer and shares_for_peer > 0:
                pe_ps = (implied_pe / shares_for_peer) if implied_pe==implied_pe else np.nan
                ev_ps = (eq_from_ev / shares_for_peer) if eq_from_ev==eq_from_ev else np.nan
                ps_ps = (implied_ps / shares_for_peer) if implied_ps==implied_ps else np.nan
            else:
                pe_ps = ev_ps = ps_ps = np.nan

            k1, k2, k3 = st.columns(3)
            k1.metric("P/E median implied Equity (per share)", f"{pe_ps:,.2f} {currency}" if pe_ps==pe_ps else "—")
            k2.metric("EV/EBITDA median implied Equity (per share)", f"{ev_ps:,.2f} {currency}" if ev_ps==ev_ps else "—")
            k3.metric("P/S median implied Equity (per share)", f"{ps_ps:,.2f} {currency}" if ps_ps==ps_ps else "—")

        except Exception:
            st.info("Not enough peer data to compute implied values.")
    else:
        st.info("No peer data could be fetched.")
else:
    st.info("Add peer tickers in the sidebar to enable comparables.")

# -------------------------- #
#     WHAT TO LOOK FOR       #
# -------------------------- #
st.header("🧭 How to Read This Company Through Its Financials")
st.markdown(
"""
**Income Statement**: Track whether revenue, gross profit, and operating income grow together. Persistent margin expansion signals operating leverage.
  
**Balance Sheet**: Check net debt (debt minus cash) and equity growth. Rising retained earnings with manageable leverage is a good sign.

**Cash Flow**: FCFE ≈ CFO − CapEx. Positive and growing FCFE supports buybacks/dividends and organic reinvestment.

**DCF**: Intrinsic value depends heavily on long-run growth and discount rate. Use the sensitivity heatmap to see a realistic range.

**Comparables**: Benchmark P/E, EV/EBITDA, and P/S vs peers. Large discounts/premiums should be explainable by growth, risk, or margins.
"""
)

st.caption("Disclaimer: Educational tool. Data can be incomplete/inaccurate. Always verify against company filings before making decisions.")
