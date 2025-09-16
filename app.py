import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import re

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

    # info
    try:
        info = t.info if hasattr(t, "info") else {}
        if not isinstance(info, dict):
            info = dict(info)
    except Exception:
        info = {}

    # fast_info -> dict
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
    if s is None or s.empty:
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

def millions(x):
    try:
        return x / 1_000_000.0
    except Exception:
        return np.nan

def format_for_display(df: pd.DataFrame, int_like: bool = True) -> pd.DataFrame:
    """
    Safely format a financial statement DataFrame into strings for display.
    - Numbers become '1,234' (0 decimals). Others stay as text.
    - Avoids Pandas Styler to prevent ValueErrors on mixed dtypes.
    """
    if df is None or df.empty:
        return df

    def fmt_cell(v):
        try:
            if pd.isna(v):
                return ""
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                return f"{float(v):,.0f}" if int_like else f"{float(v):,.2f}"
            v_num = pd.to_numeric(v)
            if pd.isna(v_num):
                return str(v)
            return f"{float(v_num):,.0f}" if int_like else f"{float(v_num):,.2f}"
        except Exception:
            return str(v)

    return df.copy().applymap(fmt_cell)

def dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

TICKER_RE = re.compile(r"^[A-Z.\-]{1,10}$")

def parse_manual_peers(text: str):
    if not text:
        return []
    raw = [p.strip().upper() for p in text.split(",") if p.strip()]
    return [p for p in raw if TICKER_RE.match(p)]

# ---------- Industry normalization & synonyms ----------
def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("&", "and")
    for ch in [",", "/", "-", "(", ")", ".", "'"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

INDUSTRY_SYNONYMS = {
    # Retail & Consumer Defensive
    "discount stores": {"discount stores", "discount retail", "general merchandise stores", "hypermarkets and super centers", "warehouse clubs and superstores"},
    "hypermarkets and super centers": {"hypermarkets and super centers", "discount stores", "general merchandise stores", "warehouse clubs and superstores"},
    "grocery stores": {"grocery stores", "supermarkets", "food retailers", "food and staples retailing"},
    "department stores": {"department stores"},
    "drug retailers": {"drug retailers", "pharmacies and drug stores"},
    "internet retail": {"internet retail", "e commerce", "internet and direct marketing retail"},
    "packaged foods": {"packaged foods", "packaged foods and meats", "processed foods"},
    "beverages non alcoholic": {"beverages non alcoholic", "non alcoholic beverages"},
    "beverages breweries": {"beverages breweries", "brewers"},
    "household and personal products": {"household and personal products"},
    # Lodging / Casinos
    "resorts and casinos": {"resorts and casinos", "casinos and gaming"},
    "hotels and motels": {"hotels and motels", "lodging"},
    # Tech buckets (kept tight)
    "consumer electronics": {"consumer electronics", "computer hardware"},
    "software infrastructure": {"software infrastructure", "software"},
    "semiconductors": {"semiconductors", "semiconductor equipment and materials"},
}

def industry_matches(anchor_industry: str, candidate_industry: str) -> bool:
    a = _norm(anchor_industry)
    c = _norm(candidate_industry)
    if not a or not c:
        return False
    if a == c:
        return True
    # synonym sets
    for key, group in INDUSTRY_SYNONYMS.items():
        if a in group and c in group:
            return True
    return False

# ---------- Auto Peer Discovery (yahooquery + robust filtering + curated fallback) ----------
@st.cache_resource(show_spinner=False)
def auto_peers_for_ticker(symbol: str, sector: str, industry: str, limit: int = 12):
    """
    Discover peers:
    1) yahooquery.search on industry & synonyms; fetch asset_profile and filter by exact/synonym industry match.
    2) If still insufficient, relax to same sector (exact).
    3) If still insufficient, use curated lists keyed by industry/sector.
    Always exclude the original symbol and dedupe.
    """
    peers = []

    # Helper: try yahooquery search with multiple query terms, then filter strictly
    def try_yq(industry_terms, sector_term, need_count):
        syms = set()
        try:
            from yahooquery import search as yq_search
            from yahooquery import Ticker as YQTicker
        except Exception:
            return []

        # Search by each industry term
        try:
            for term in industry_terms:
                if not term:
                    continue
                res = yq_search(term, first=80)
                if isinstance(res, dict) and "quotes" in res:
                    for q in res["quotes"]:
                        s = q.get("symbol")
                        qt = str(q.get("quoteType", "")).upper()
                        if not s or qt not in ("EQUITY", "ETF", "MUTUALFUND"):
                            continue
                        s = s.upper()
                        if TICKER_RE.match(s):
                            syms.add(s)
        except Exception:
            pass

        if not syms:
            return []

        # Fetch profiles and filter
        try:
            yq = YQTicker(list(syms)[:150], asynchronous=True)
            prof = yq.asset_profile
        except Exception:
            prof = {}

        same_ind = []
        same_sector = []
        for s, dat in prof.items():
            if not isinstance(dat, dict):
                continue
            cand_ind = str(dat.get("industry", ""))
            cand_sec = str(dat.get("sector", ""))
            if industry and industry_matches(industry, cand_ind):
                same_ind.append(s)
            elif sector_term and _norm(cand_sec) == _norm(sector_term):
                same_sector.append(s)

        # Prefer industry matches, fall back to sector
        out = [x for x in same_ind if x != symbol]
        if len(out) < need_count:
            # augment with sector matches (excluding anything with clearly different industries like tech when anchor is consumer defensive)
            out += [x for x in same_sector if x != symbol and x not in out]
        return out[:limit]

    # 1) Strict by industry (with synonyms)
    industry_terms = [industry]
    # add synonyms for broader capture
    for k, group in INDUSTRY_SYNONYMS.items():
        if _norm(industry) in group:
            industry_terms.extend(group)
            break

    found = try_yq(industry_terms, sector, need_count=limit)
    if found:
        peers = found

    # 2) If still thin, try sector-only pass (kept tight)
    if not peers or len(peers) < 3:
        sector_pass = try_yq([sector], sector, need_count=limit)
        # Ensure sector-only additions actually share sector exactly
        sector_pass = [p for p in sector_pass if p not in peers]
        peers += sector_pass
        peers = peers[:limit]

    # 3) Curated fallback (industry-first, then sector)
    if not peers or len(peers) < 3:
        peers += curated_fallback(symbol, sector, industry)
        peers = dedupe_keep_order(peers)[:limit]

    # basic clean-up
    peers = [p for p in peers if TICKER_RE.match(p) and p != symbol]
    return dedupe_keep_order(peers)[:limit]

def curated_fallback(symbol: str, sector: str, industry: str) -> list:
    """Curated lists for common categories to avoid unrelated picks."""
    aind = _norm(industry)
    asec = _norm(sector)
    # Retail / Consumer Defensive specifics
    if industry_matches("discount stores", aind) or industry_matches("hypermarkets and super centers", aind):
        return [x for x in ["COST", "TGT", "DG", "DLTR", "BJ"] if x != symbol]
    if industry_matches("grocery stores", aind):
        return [x for x in ["KR", "ACI", "SFM", "GO", "IMKTA", "CASY"] if x != symbol]
    if industry_matches("drug retailers", aind):
        return [x for x in ["WBA", "CVS", "RAD"] if x != symbol]
    if industry_matches("internet retail", aind):
        return [x for x in ["AMZN", "SHOP", "MELI", "BABA"] if x != symbol]
    if industry_matches("packaged foods", aind):
        return [x for x in ["GIS", "K", "KHC", "CAG", "MDLZ", "HSY"] if x != symbol]
    if industry_matches("beverages non alcoholic", aind):
        return [x for x in ["KO", "PEP", "KDP", "MNST"] if x != symbol]
    if industry_matches("household and personal products", aind):
        return [x for x in ["PG", "CL", "KMB", "CLX", "CHD"] if x != symbol]

    # Lodging / Casinos
    if industry_matches("resorts and casinos", aind):
        return [x for x in ["MGM", "LVS", "CZR", "MLCO", "PENN", "BALY"] if x != symbol]
    if industry_matches("hotels and motels", aind):
        return [x for x in ["MAR", "HLT", "H", "IHG", "HGV"] if x != symbol]

    # Tech (keep aligned)
    if industry_matches("semiconductors", aind):
        return [x for x in ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU"] if x != symbol]
    if industry_matches("software infrastructure", aind):
        return [x for x in ["MSFT", "ADBE", "ORCL", "CRM", "NOW", "INTU"] if x != symbol]
    if industry_matches("consumer electronics", aind):
        return [x for x in ["AAPL", "HPQ", "DELL", "SSNLF"] if x != symbol]

    # Sector-level guardrails (last resort; avoids tech for WMT etc.)
    if asec == _norm("Consumer Defensive"):
        return [x for x in ["COST", "TGT", "KR", "DG", "DLTR", "BJ"] if x != symbol]
    if asec == _norm("Consumer Cyclical"):
        return [x for x in ["HD", "LOW", "ROST", "TJX", "M", "KSS"] if x != symbol]
    if asec == _norm("Technology"):
        return [x for x in ["MSFT", "GOOGL", "NVDA", "META", "ORCL", "ADBE"] if x != symbol]
    if asec == _norm("Communication Services"):
        return [x for x in ["GOOGL", "META", "NFLX", "TTWO", "EA"] if x != symbol]
    if asec == _norm("Healthcare"):
        return [x for x in ["JNJ", "PFE", "MRK", "ABBV", "LLY"] if x != symbol]
    if asec == _norm("Energy"):
        return [x for x in ["XOM", "CVX", "COP", "SLB", "HAL"] if x != symbol]

    # Very last: nothing unrelated
    return []

# -------------------------- #
#           SIDEBAR          #
# -------------------------- #
st.sidebar.title("🔎 Stock & Settings")
ticker = st.sidebar.text_input("Primary Ticker (e.g., AAPL, MSFT, TSLA):", value="AAPL").upper().strip()

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
st.sidebar.caption("Tip: Add or remove peers in the Comparables section to refine the set.")
st.sidebar.markdown(
    '[👤 Built by **Navjot Dhah**](https://www.linkedin.com/in/navjot-dhah-57870b238)',
    unsafe_allow_html=True
)

# -------------------------- #
#         MAIN HEADER        #
# -------------------------- #
st.title("📊 Company Analyzer & Valuation ")
# Top-right credit with LinkedIn
st.markdown(
    '<div style="text-align:right;">'
    'Built by <a href="https://www.linkedin.com/in/navjot-dhah-57870b238" target="_blank">Navjot Dhah</a>'
    '</div>',
    unsafe_allow_html=True
)
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
            st.dataframe(format_for_display(annual_df), use_container_width=True, height=360)
        else:
            st.info("Annual data unavailable.")
    with c2:
        if quarterly_df is not None and not quarterly_df.empty:
            st.caption("Quarterly")
            st.dataframe(format_for_display(quarterly_df), use_container_width=True, height=360)
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

    df_proj = pd.DataFrame({"Year": years, "FCFE": proj, "Discount Factor": disc, "PV": np.array(proj)*disc})
    fcfe_fig = px.bar(df_proj, x="Year", y="FCFE", title="Projected FCFE")
    pv_fig = px.bar(df_proj, x="Year", y="PV", title="Present Value of FCFE")
    st.plotly_chart(fcfe_fig, use_container_width=True)
    st.plotly_chart(pv_fig, use_container_width=True)

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
st.header("📈 Comparables ")

# Auto-detect peers for current ticker — now strictly same industry, then sector; curated fallback avoids unrelated picks
auto_suggestions = auto_peers_for_ticker(
    ticker,
    sector if sector and sector != "—" else "",
    industry if industry and industry != "—" else ""
)

colp1, colp2 = st.columns([2, 1])
with colp1:
    st.caption("Suggested peers (same industry preferred; sector as fallback). Unselect any you don't want.")
    selected_auto = st.multiselect(
        "Auto-detected peers",
        options=auto_suggestions,
        default=auto_suggestions[: min(6, len(auto_suggestions))]
    )
with colp2:
    manual_peers_text = st.text_input("Add custom peers (comma-separated)", value="")
    manual_peers = parse_manual_peers(manual_peers_text)

# Final peer list (unique, ordered)
peers = [p for p in dedupe_keep_order((selected_auto or []) + (manual_peers or [])) if p != ticker]

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
        st.dataframe(comp_df.applymap(
            lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else v
        ), use_container_width=True)

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

            base = comp_df.loc[ticker]
            implied_pe = (med_pe * base["Net Income (TTM)"]) if med_pe==med_pe else np.nan
            implied_ev = (med_ev_ebitda * base["EBITDA (TTM)"]) if med_ev_ebitda==med_ev_ebitda else np.nan
            implied_ps = (med_ps * base["Revenue (TTM)"]) if med_ps==med_ps else np.nan

            net_debt = float(base["EV"] - base["Market Cap"]) if base["EV"]==base["EV"] and base["Market Cap"]==base["Market Cap"] else np.nan
            eq_from_ev = implied_ev - net_debt if implied_ev==implied_ev and net_debt==net_debt else np.nan

            shares_for_peer = base.get("Shares", np.nan)
            if isinstance(shares_for_peer, (int, float, np.integer, np.floating)) and shares_for_peer and shares_for_peer > 0:
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
    st.info("Select peers from the auto-suggestions or add custom tickers above to enable comparables.")

# -------------------------- #
#     WHAT TO LOOK FOR       #
# -------------------------- #
st.header("🧭 Basics Of Reading Financials ")
st.markdown(
"""
**Income Statement**: Track whether revenue, gross profit, and operating income grow together. Persistent margin expansion signals operating leverage.
  
**Balance Sheet**: Check net debt (debt minus cash) and equity growth. Rising retained earnings with manageable leverage is a good sign.

**Cash Flow**: FCFE ≈ CFO − CapEx. Positive and growing FCFE supports buybacks/dividends and organic reinvestment.

**DCF**: Intrinsic value depends heavily on long-run growth and discount rate. Use the sensitivity heatmap to see a realistic range.

**Comparables**: Benchmark P/E, EV/EBITDA, and P/S vs peers. Large discounts/premiums should be explainable by growth, risk, or margins.
"""
)

st.caption("Disclaimer: Educational tool. Data can be incomplete/inaccurate. Always verify against company filings. ")
