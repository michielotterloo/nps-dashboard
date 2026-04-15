import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
import requests
import json
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dutchview NPS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URLS = {
    "EDC": "https://web.edcontrols.com/api/v1/nps/report?product=EDCONTROLS",
    "FW": "https://dutchview.flexwhere.com/api/v1/nps/report?product=FLEXWHERE",
}
LABELS = {"EDC": "Ed Controls", "FW": "Flexwhere"}
NPS_TARGETS = {"EDC": 50, "FW": 32}
EXCEL_PATH = Path(__file__).parent / "data" / "NPS Dutchview.xlsx"
EXCLUDED_DOMAINS = {"dutchview.com", "mailinator.com"}
GENERIC_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "outlook.com", "live.nl", "live.com",
    "yahoo.com", "icloud.com", "me.com", "msn.com", "ziggo.nl",
    "kpnmail.nl", "xs4all.nl", "planet.nl", "hetnet.nl", "home.nl",
    "upcmail.nl", "casema.nl", "chello.nl", "quicknet.nl", "tele2.nl",
    "online.nl", "solcon.nl", "zonnet.nl", "versatel.nl", "wxs.nl",
    "gmail.de", "web.de", "gmx.de", "gmx.net", "t-online.de",
    "freenet.de", "arcor.de", "yahoo.de", "hotmail.de", "outlook.de",
    "googlemail.com",
}
CUSTOMER_LOOKUP_PATH = Path(__file__).parent / "data" / "customer_lookup.json"


@st.cache_data(ttl=3600)
def load_customer_lookup():
    """Load domain → customer name/owner mapping from pre-built lookup."""
    if not CUSTOMER_LOOKUP_PATH.exists():
        return {}
    with open(CUSTOMER_LOOKUP_PATH) as f:
        return json.load(f)


def classify_domain(domain, customer_lookup):
    """Classify a domain into: Klant (in CRM), Particulier (generic email), Overig."""
    d = str(domain).lower().strip() if pd.notna(domain) else ""
    if d and d in customer_lookup:
        return "Klant"
    elif d in GENERIC_EMAIL_DOMAINS:
        return "Particulier"
    else:
        return "Overig"


def domain_to_name(domain, customer_lookup):
    """Convert domain to customer name if known."""
    if pd.isna(domain):
        return "Onbekend"
    d = str(domain).lower().strip()
    entry = customer_lookup.get(d)
    if entry:
        return entry.get("name") or domain
    return domain


@st.cache_data(ttl=3600)
def fetch_api_data(prefix):
    """Fetch live NPS data from API. Returns DataFrame or None on failure."""
    try:
        r = requests.get(API_URLS[prefix], timeout=30)
        r.raise_for_status()
        df = pd.read_excel(BytesIO(r.content))
        df = df.rename(columns=lambda c: c.strip())
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")
        # Extract domain from EMAIL
        if prefix == "FW":
            # Flexwhere format: "name/domain.com"
            df["DOMAIN"] = df["EMAIL"].astype(str).str.extract(r"/([^/]+)$")[0]
        else:
            # EdControls format: "user@domain.com"
            df["DOMAIN"] = df["EMAIL"].astype(str).str.extract(r"@(.+)$")[0]
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_historical_data(prefix):
    """Load historical data from Parquet (fast) or Excel (fallback)."""
    parquet_path = Path(__file__).parent / "data" / f"{prefix}_responses.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if not EXCEL_PATH.exists():
        return None
    xls = pd.ExcelFile(EXCEL_PATH)
    resp = pd.read_excel(xls, sheet_name=f"{prefix} responses", header=0,
                         usecols=["EMAIL", "PLATFORM", "SCORE", "MESSAGE", "DATE", "DOMAIN"])
    resp = resp.rename(columns=lambda c: c.strip())
    resp["DATE"] = pd.to_datetime(resp["DATE"], errors="coerce")
    resp["SCORE"] = pd.to_numeric(resp["SCORE"], errors="coerce")
    if "DOMAIN" not in resp.columns or resp["DOMAIN"].isna().all():
        if prefix == "FW":
            resp["DOMAIN"] = resp["EMAIL"].astype(str).str.extract(r"/([^/]+)$")[0]
        else:
            resp["DOMAIN"] = resp["EMAIL"].astype(str).str.extract(r"@(.+)$")[0]
    return resp


@st.cache_data(ttl=3600)
def load_product(prefix):
    """Load and merge API + Excel data for a product."""
    api_df = fetch_api_data(prefix)
    excel_df = load_historical_data(prefix)

    if api_df is not None and excel_df is not None:
        api_min_date = api_df["DATE"].min()
        historical = excel_df[excel_df["DATE"] < api_min_date].copy()
        resp = pd.concat([historical, api_df], ignore_index=True)
    elif api_df is not None:
        resp = api_df
    elif excel_df is not None:
        resp = excel_df
    else:
        return None

    resp = resp.drop_duplicates(subset=["EMAIL", "DATE"], keep="last")
    resp = resp[~resp["DOMAIN"].str.lower().isin(EXCLUDED_DOMAINS)]
    resp = resp.sort_values("DATE").reset_index(drop=True)

    # Enrich with customer names from HubSpot
    customer_lookup = load_customer_lookup()
    resp["CUSTOMER"] = resp["DOMAIN"].apply(lambda d: domain_to_name(d, customer_lookup))
    resp["DOMAIN_TYPE"] = resp["DOMAIN"].apply(lambda d: classify_domain(d, customer_lookup))
    resp["OWNER"] = resp["DOMAIN"].apply(
        lambda d: (customer_lookup.get(str(d).lower().strip()) or {}).get("owner") or "" if pd.notna(d) else ""
    )

    # Add time columns
    resp["MONTH"] = resp["DATE"].dt.to_period("M").astype(str)
    resp["WEEK"] = resp["DATE"].dt.strftime("%G-W%V")
    resp["QUARTER"] = resp["DATE"].dt.to_period("Q").astype(str)

    # Classify NPS category
    resp["NPS_CAT"] = pd.cut(
        resp["SCORE"],
        bins=[-1, 5, 7, 10],
        labels=["Detractor (0-5)", "Passive (6-7)", "Promoter (8-10)"],
    )

    # Build monthly aggregates from responses
    monthly = build_monthly(resp)
    weekly = build_weekly(resp)
    customers = build_customers(resp)

    return {
        "label": LABELS[prefix],
        "responses": resp,
        "monthly": monthly,
        "weekly": weekly,
        "customers": customers,
        "customer_lookup": customer_lookup,
    }


def calc_nps(scores):
    scores = scores.dropna()
    if len(scores) == 0:
        return 0
    promoters = (scores >= 8).sum()
    detractors = (scores <= 5).sum()
    return (promoters - detractors) / len(scores) * 100


def build_monthly(resp):
    scored = resp.dropna(subset=["SCORE"])
    monthly = scored.groupby("MONTH").agg(
        Responses=("SCORE", "count"),
        Month_NPS=("SCORE", calc_nps),
    ).reset_index().rename(columns={"MONTH": "Month_Label"})
    monthly = monthly.sort_values("Month_Label").reset_index(drop=True)

    # Quarter NPS
    scored_q = scored.copy()
    scored_q["Q"] = scored_q["DATE"].dt.to_period("Q").astype(str)
    q_nps = scored_q.groupby("Q")["SCORE"].apply(calc_nps).to_dict()
    monthly["Quarter_NPS"] = monthly["Month_Label"].apply(
        lambda m: q_nps.get(pd.Period(m, "M").asfreq("Q").strftime("%YQ%q"), None)
    )

    # MAT (Moving Annual Total) — rolling 12-month NPS
    mat_values = []
    months_list = monthly["Month_Label"].tolist()
    for i, m in enumerate(months_list):
        start_idx = max(0, i - 11)
        window_months = set(months_list[start_idx : i + 1])
        window_data = scored[scored["MONTH"].isin(window_months)]
        mat_values.append(calc_nps(window_data["SCORE"]))
    monthly["MAT_NPS"] = mat_values

    return monthly


def build_weekly(resp):
    scored = resp.dropna(subset=["SCORE"])
    weekly = scored.groupby("WEEK").agg(
        Responses=("SCORE", "count"),
        Week_NPS=("SCORE", calc_nps),
    ).reset_index().rename(columns={"WEEK": "Week_Label"})
    weekly = weekly.sort_values("Week_Label").reset_index(drop=True)
    return weekly


def build_customers(resp):
    cust_agg = (
        resp.dropna(subset=["SCORE", "CUSTOMER"])
        .groupby("CUSTOMER")
        .agg(
            DOMAIN=("DOMAIN", "first"),
            DOMAIN_TYPE=("DOMAIN_TYPE", "first"),
            OWNER=("OWNER", "first"),
            Responses=("SCORE", "count"),
            Avg_Score=("SCORE", "mean"),
            Promoters=("NPS_CAT", lambda x: (x == "Promoter (8-10)").sum()),
            Detractors=("NPS_CAT", lambda x: (x == "Detractor (0-5)").sum()),
        )
        .reset_index()
    )
    cust_agg["NPS"] = (
        (cust_agg["Promoters"] - cust_agg["Detractors"]) / cust_agg["Responses"] * 100
    )
    return cust_agg.sort_values("Responses", ascending=False)


def nps_color(nps):
    if nps >= 50:
        return "#22c55e"
    elif nps >= 30:
        return "#84cc16"
    elif nps >= 0:
        return "#f59e0b"
    else:
        return "#ef4444"


def metric_card(label, value, suffix="", delta=None, delta_label=""):
    color = nps_color(value) if suffix == "" else "#3b82f6"
    delta_html = ""
    if delta is not None:
        d_color = "#22c55e" if delta >= 0 else "#ef4444"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="font-size:0.75rem;color:{d_color};white-space:nowrap">{arrow} {delta:+.1f} {delta_label}</div>'
    val_str = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    st.markdown(
        f"""
        <div style="background:#1e293b;border-radius:12px;padding:1rem;text-align:center;border-left:4px solid {color}">
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.2rem;white-space:nowrap">{label}</div>
            <div style="font-size:1.8rem;font-weight:700;color:{color};white-space:nowrap">{val_str}{suffix}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- MAIN ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { background: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Dutchview NPS")
product = st.sidebar.radio("Product", ["EDC", "FW"], format_func=lambda x: LABELS[x])

data = load_product(product)
if data is None:
    st.error("Kan geen data laden. Controleer de API verbinding en/of het Excel bestand.")
    st.stop()

resp = data["responses"]
monthly = data["monthly"]
weekly = data["weekly"]
customers = data["customers"]

# Data source indicator
api_df = fetch_api_data(product)
st.sidebar.markdown("---")
if api_df is not None:
    api_latest = api_df["DATE"].max().strftime("%d-%m-%Y %H:%M")
    st.sidebar.success(f"Live data t/m {api_latest}")
else:
    st.sidebar.warning("API offline — Excel data wordt gebruikt")

months_available = sorted(resp["MONTH"].dropna().unique())
# Default to last 12 months (MAT window)
default_start = months_available[-12] if len(months_available) >= 12 else months_available[0]
date_range = st.sidebar.select_slider(
    "Periode",
    options=months_available,
    value=(default_start, months_available[-1]),
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🎯 NPS Target: ≥ {NPS_TARGETS[product]}**")
if data is not None and len(data["monthly"]) > 0:
    _mat = data["monthly"].iloc[-1]["MAT_NPS"]
    _delta = _mat - NPS_TARGETS[product]
    _icon = "✅" if _delta >= 0 else "⚠️"
    st.sidebar.markdown(f"MAT NPS: **{_mat:.1f}** {_icon}")

if st.sidebar.button("Ververs data"):
    st.cache_data.clear()
    st.rerun()

# Filter responses by date range
resp_filtered = resp[(resp["MONTH"] >= date_range[0]) & (resp["MONTH"] <= date_range[1])]
scored = resp_filtered.dropna(subset=["SCORE"])

# --- HEADER ---
st.title(f"NPS Dashboard — {data['label']}")

# --- KPI ROW ---
current_nps = calc_nps(scored["SCORE"])
prev_month = months_available[-2] if len(months_available) > 1 else months_available[0]
prev_scored = resp[resp["MONTH"] == prev_month].dropna(subset=["SCORE"])
prev_nps = calc_nps(prev_scored["SCORE"])
delta_nps = current_nps - prev_nps

# MAT NPS from monthly data
mat_nps = monthly.iloc[-1]["MAT_NPS"] if len(monthly) > 0 else current_nps
target = NPS_TARGETS[product]
target_delta = mat_nps - target

total_responses = len(resp_filtered)
total_scored = len(scored)
response_rate = total_scored / total_responses * 100 if total_responses > 0 else 0
avg_score = scored["SCORE"].mean() if len(scored) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    metric_card("NPS Score", current_nps, delta=delta_nps, delta_label="vs vorige maand")
with col2:
    metric_card("MAT NPS", mat_nps, delta=target_delta, delta_label=f"vs target ({target})")
with col3:
    metric_card("Gem. Score", avg_score, suffix="/10")
with col4:
    metric_card("Responses", float(total_scored), suffix="")
with col5:
    metric_card("Response Rate", response_rate, suffix="%")

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab5, tab4 = st.tabs(["Trend", "Verdeling", "Klanten", "Inzichten", "Responses"])

with tab1:
    view = st.radio("Weergave", ["Maand", "Week"], horizontal=True, key="trend_view")

    if view == "Maand":
        df_plot = monthly[monthly["Month_Label"].between(date_range[0], date_range[1])].copy()
        if len(df_plot) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot["Month_Label"], y=df_plot["Month_NPS"],
                name="Maand NPS", mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=df_plot["Month_Label"], y=df_plot["Quarter_NPS"],
                name="Kwartaal NPS", mode="lines",
                line=dict(color="#8b5cf6", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=df_plot["Month_Label"], y=df_plot["MAT_NPS"],
                name="MAT NPS", mode="lines",
                line=dict(color="#f59e0b", width=3),
            ))
            fig.update_layout(
                title="NPS Trend per Maand",
                xaxis_title="Maand", yaxis_title="NPS Score",
                yaxis=dict(range=[-100, 100]),
                template="plotly_dark", height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            # Target line
            target = NPS_TARGETS[product]
            fig.add_hline(
                y=target, line_dash="dash", line_color="#22c55e", opacity=0.7,
                annotation_text=f"Target ≥ {target}",
                annotation_position="top left",
                annotation_font_color="#22c55e",
            )
            st.plotly_chart(fig, width="stretch")

            col_a, col_b = st.columns(2)
            with col_a:
                fig_resp = px.bar(
                    df_plot, x="Month_Label", y="Responses",
                    title="Aantal Responses per Maand",
                    template="plotly_dark",
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_resp.update_layout(height=300, xaxis_title="Maand", yaxis_title="Responses")
                st.plotly_chart(fig_resp, width="stretch")
            with col_b:
                if len(df_plot) > 1:
                    latest_mat = df_plot.iloc[-1]["MAT_NPS"]
                    latest_month = df_plot.iloc[-1]["Month_NPS"]
                    latest_quarter = df_plot.iloc[-1]["Quarter_NPS"]
                    target = NPS_TARGETS[product]
                    mat_delta = latest_mat - target
                    mat_color = "#22c55e" if mat_delta >= 0 else "#ef4444"
                    mat_arrow = "✅" if mat_delta >= 0 else "⚠️"
                    st.markdown(f"""
                    **MAT NPS**: {latest_mat:.1f} {mat_arrow} (target: {target})

                    **Laatste maand NPS**: {latest_month:.1f}

                    **Laatste kwartaal NPS**: {latest_quarter:.1f}
                    """)

                    # YoY comparison
                    latest_month_label = df_plot.iloc[-1]["Month_Label"]
                    yoy_month = latest_month_label[:4]
                    yoy_target = str(int(yoy_month) - 1) + latest_month_label[4:]
                    yoy_row = df_plot[df_plot["Month_Label"] == yoy_target]
                    if len(yoy_row) > 0:
                        yoy_nps = yoy_row.iloc[0]["Month_NPS"]
                        yoy_delta = latest_month - yoy_nps
                        yoy_color = "#22c55e" if yoy_delta >= 0 else "#ef4444"
                        yoy_arrow = "▲" if yoy_delta >= 0 else "▼"
                        st.markdown(
                            f'<div style="background:#1e293b;border-radius:8px;padding:0.8rem;margin-top:0.5rem">'
                            f'<span style="color:#94a3b8;font-size:0.8rem">Year-over-Year ({yoy_target})</span><br>'
                            f'<span style="color:{yoy_color};font-size:1.2rem;font-weight:700">'
                            f'{yoy_arrow} {yoy_delta:+.1f}</span>'
                            f'<span style="color:#94a3b8;font-size:0.8rem"> ({yoy_nps:.1f} → {latest_month:.1f})</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
    else:
        df_plot = weekly.copy()
        # Filter weeks within date range
        month_start = date_range[0]
        month_end = date_range[1]
        week_months = resp.dropna(subset=["WEEK", "MONTH"]).groupby("WEEK")["MONTH"].first()
        valid_weeks = week_months[(week_months >= month_start) & (week_months <= month_end)].index
        df_plot = df_plot[df_plot["Week_Label"].isin(valid_weeks)]
        if len(df_plot) == 0:
            df_plot = weekly.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["Week_Label"], y=df_plot["Week_NPS"],
            name="Week NPS", mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
        ))
        fig.update_layout(
            title="NPS Trend per Week",
            xaxis_title="Week", yaxis_title="NPS Score",
            yaxis=dict(range=[-100, 100]),
            template="plotly_dark", height=450,
            hovermode="x unified",
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        target = NPS_TARGETS[product]
        fig.add_hline(
            y=target, line_dash="dash", line_color="#22c55e", opacity=0.7,
            annotation_text=f"Target ≥ {target}",
            annotation_position="top left",
            annotation_font_color="#22c55e",
        )
        st.plotly_chart(fig, width="stretch")

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        cat_counts = scored["NPS_CAT"].value_counts()
        colors_map = {
            "Promoter (8-10)": "#22c55e",
            "Passive (6-7)": "#f59e0b",
            "Detractor (0-5)": "#ef4444",
        }
        fig_pie = px.pie(
            names=cat_counts.index,
            values=cat_counts.values,
            title="NPS Verdeling",
            color=cat_counts.index,
            color_discrete_map=colors_map,
            template="plotly_dark",
        )
        fig_pie.update_traces(textinfo="percent+label", textposition="inside")
        fig_pie.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_pie, width="stretch")

    with col2:
        score_dist = scored["SCORE"].value_counts().sort_index().reset_index()
        score_dist.columns = ["Score", "Count"]
        score_dist["Color"] = score_dist["Score"].apply(
            lambda s: "#ef4444" if s <= 6 else ("#f59e0b" if s <= 8 else "#22c55e")
        )
        fig_bar = px.bar(
            score_dist, x="Score", y="Count",
            title="Score Verdeling (0-10)",
            color="Color",
            color_discrete_map="identity",
            template="plotly_dark",
        )
        fig_bar.update_layout(height=400, showlegend=False, xaxis=dict(dtick=1))
        st.plotly_chart(fig_bar, width="stretch")

    if "PLATFORM" in scored.columns:
        platform_nps = scored.groupby("PLATFORM")["SCORE"].apply(calc_nps).reset_index()
        platform_nps.columns = ["Platform", "NPS"]
        platform_nps = platform_nps.sort_values("NPS", ascending=True)
        fig_plat = px.bar(
            platform_nps, x="NPS", y="Platform", orientation="h",
            title="NPS per Platform",
            template="plotly_dark",
            color="NPS",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        )
        fig_plat.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_plat, width="stretch")

    monthly_cat = (
        scored.groupby(["MONTH", "NPS_CAT"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if len(monthly_cat) > 0:
        monthly_cat_pct = monthly_cat.copy()
        cat_cols = [c for c in monthly_cat_pct.columns if c != "MONTH"]
        for c in cat_cols:
            monthly_cat_pct[c] = monthly_cat_pct[c] / monthly_cat_pct[cat_cols].sum(axis=1) * 100

        fig_area = go.Figure()
        for cat, color in colors_map.items():
            if cat in monthly_cat_pct.columns:
                fig_area.add_trace(go.Scatter(
                    x=monthly_cat_pct["MONTH"], y=monthly_cat_pct[cat],
                    name=cat, stackgroup="one", fillcolor=color,
                    line=dict(color=color, width=0.5),
                ))
        fig_area.update_layout(
            title="NPS Categorie Verdeling over Tijd (%)",
            template="plotly_dark", height=400,
            yaxis_title="Percentage", xaxis_title="Maand",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_area, width="stretch")

with tab3:
    # Domain type filter
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        domain_type_filter = st.multiselect(
            "Type",
            ["Klant", "Overig", "Particulier"],
            default=["Klant", "Overig"],
            key="domain_type_filter",
        )
    with filter_col2:
        min_responses = st.slider("Minimum responses", 1, 100, 10, key="min_resp")

    cust_filtered = customers[
        (customers["Responses"] >= min_responses) &
        (customers["DOMAIN_TYPE"].isin(domain_type_filter))
    ].copy()

    # Show summary counts per type
    type_counts = customers.groupby("DOMAIN_TYPE").agg(
        Klanten=("CUSTOMER", "count"),
        Responses=("Responses", "sum"),
    ).reindex(["Klant", "Overig", "Particulier"]).fillna(0).astype(int)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        k = type_counts.loc["Klant"] if "Klant" in type_counts.index else pd.Series({"Klanten": 0, "Responses": 0})
        st.markdown(f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem;text-align:center;border-left:3px solid #3b82f6"><span style="color:#94a3b8;font-size:0.75rem">Klanten (CRM)</span><br><span style="font-weight:700;color:#3b82f6">{k["Klanten"]}</span> <span style="color:#64748b;font-size:0.75rem">({k["Responses"]} responses)</span></div>', unsafe_allow_html=True)
    with sc2:
        o = type_counts.loc["Overig"] if "Overig" in type_counts.index else pd.Series({"Klanten": 0, "Responses": 0})
        st.markdown(f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem;text-align:center;border-left:3px solid #f59e0b"><span style="color:#94a3b8;font-size:0.75rem">Overig (niet in CRM)</span><br><span style="font-weight:700;color:#f59e0b">{o["Klanten"]}</span> <span style="color:#64748b;font-size:0.75rem">({o["Responses"]} responses)</span></div>', unsafe_allow_html=True)
    with sc3:
        p = type_counts.loc["Particulier"] if "Particulier" in type_counts.index else pd.Series({"Klanten": 0, "Responses": 0})
        st.markdown(f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem;text-align:center;border-left:3px solid #64748b"><span style="color:#94a3b8;font-size:0.75rem">Particulier (gmail etc.)</span><br><span style="font-weight:700;color:#64748b">{p["Klanten"]}</span> <span style="color:#64748b;font-size:0.75rem">({p["Responses"]} responses)</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        top_15 = cust_filtered.nlargest(15, "NPS")
        fig_top = px.bar(
            top_15, x="NPS", y="CUSTOMER", orientation="h",
            title=f"Top 15 Klanten (min. {min_responses} responses)",
            template="plotly_dark",
            color="NPS",
            color_continuous_scale=["#f59e0b", "#22c55e"],
            text="NPS",
        )
        fig_top.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_top.update_layout(height=500, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, width="stretch")

    with col2:
        bottom_15 = cust_filtered.nsmallest(15, "NPS")
        fig_bot = px.bar(
            bottom_15, x="NPS", y="CUSTOMER", orientation="h",
            title=f"Bottom 15 Klanten (min. {min_responses} responses)",
            template="plotly_dark",
            color="NPS",
            color_continuous_scale=["#ef4444", "#f59e0b"],
            text="NPS",
        )
        fig_bot.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_bot.update_layout(height=500, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bot, width="stretch")

    st.subheader("Alle Klanten")
    cust_display = cust_filtered[["CUSTOMER", "DOMAIN_TYPE", "OWNER", "Responses", "NPS", "Avg_Score", "Promoters", "Detractors"]].copy()
    cust_display["NPS"] = cust_display["NPS"].round(1)
    cust_display["Avg_Score"] = cust_display["Avg_Score"].round(2)
    cust_display = cust_display.rename(columns={
        "CUSTOMER": "Klant", "DOMAIN_TYPE": "Type", "OWNER": "Account Manager",
        "Avg_Score": "Gem. Score",
    })

    search_cust = st.text_input("Zoek klant", key="search_cust")
    if search_cust:
        cust_display = cust_display[cust_display["Klant"].str.contains(search_cust, case=False, na=False)]

    st.dataframe(cust_display.reset_index(drop=True), width="stretch", height=400)

    # --- Klant detail drill-down ---
    st.markdown("---")
    st.subheader("Klant Detail")
    customer_list = cust_filtered["CUSTOMER"].tolist()
    selected_customer = st.selectbox(
        "Selecteer een klant",
        options=[""] + customer_list,
        format_func=lambda x: "Kies een klant..." if x == "" else x,
        key="selected_customer",
    )

    if selected_customer:
        cust_resp = resp_filtered[resp_filtered["CUSTOMER"] == selected_customer].copy()
        cust_scored = cust_resp.dropna(subset=["SCORE"])

        # Show account manager if known
        cust_row = cust_filtered[cust_filtered["CUSTOMER"] == selected_customer].iloc[0]
        owner = cust_row.get("OWNER", "")
        dtype = cust_row.get("DOMAIN_TYPE", "")
        domain = cust_row.get("DOMAIN", "")
        info_parts = []
        if dtype:
            info_parts.append(f"**Type:** {dtype}")
        if owner:
            info_parts.append(f"**Account Manager:** {owner}")
        if domain:
            info_parts.append(f"**Domein:** {domain}")
        if info_parts:
            st.markdown(" · ".join(info_parts))

        # KPI row for selected customer
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("NPS", calc_nps(cust_scored["SCORE"]) if len(cust_scored) > 0 else 0)
        with c2:
            metric_card("Gem. Score", cust_scored["SCORE"].mean() if len(cust_scored) > 0 else 0, suffix="/10")
        with c3:
            metric_card("Responses", float(len(cust_scored)), suffix="")
        with c4:
            prom = (cust_scored["NPS_CAT"] == "Promoter (8-10)").sum() if len(cust_scored) > 0 else 0
            detr = (cust_scored["NPS_CAT"] == "Detractor (0-5)").sum() if len(cust_scored) > 0 else 0
            metric_card("Promoters / Detractors", float(prom), suffix=f" / {detr}")

        # NPS trend for this customer over time
        cust_monthly = cust_scored.groupby("MONTH")["SCORE"].apply(calc_nps).reset_index()
        cust_monthly.columns = ["Maand", "NPS"]
        if len(cust_monthly) > 1:
            fig_cust = go.Figure()
            fig_cust.add_trace(go.Scatter(
                x=cust_monthly["Maand"], y=cust_monthly["NPS"],
                mode="lines+markers", name="NPS",
                line=dict(color="#3b82f6", width=2),
            ))
            fig_cust.update_layout(
                title=f"NPS Trend — {selected_customer}",
                template="plotly_dark", height=300,
                xaxis_title="Maand", yaxis_title="NPS",
                yaxis=dict(range=[-100, 100]),
                hovermode="x unified",
            )
            fig_cust.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig_cust.add_hline(
                y=NPS_TARGETS[product], line_dash="dash", line_color="#22c55e", opacity=0.7,
                annotation_text=f"Target ≥ {NPS_TARGETS[product]}",
                annotation_position="top left",
                annotation_font_color="#22c55e",
            )
            st.plotly_chart(fig_cust, width="stretch")

        # Responses table
        st.markdown(f"**Responses van {selected_customer}**")
        cust_show = cust_resp[["DATE", "SCORE", "NPS_CAT", "MESSAGE", "PLATFORM"]].sort_values("DATE", ascending=False)
        cust_show = cust_show.rename(columns={
            "DATE": "Datum", "SCORE": "Score", "NPS_CAT": "Categorie",
            "MESSAGE": "Feedback", "PLATFORM": "Platform",
        })
        if "Feedback" in cust_show.columns:
            cust_show["Feedback"] = cust_show["Feedback"].astype(str).replace("nan", "")
        st.dataframe(cust_show.reset_index(drop=True), width="stretch", height=400)

with tab5:
    st.subheader("Inzichten — Recente Feedback")
    st.caption("Uitgebreide reacties (min. 20 tekens) van de afgelopen periode, gesorteerd op datum.")

    feedback = scored[scored["MESSAGE"].notna()].copy()
    feedback["MSG_STR"] = feedback["MESSAGE"].astype(str)
    feedback = feedback[feedback["MSG_STR"].str.len() >= 20]
    feedback = feedback.sort_values("DATE", ascending=False)

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("### 🟢 Positieve Feedback")
        st.caption("Promoters (score 8-10)")
        pos = feedback[feedback["SCORE"] >= 8].head(15)
        if len(pos) == 0:
            st.info("Geen uitgebreide positieve feedback in deze periode.")
        else:
            for _, r in pos.iterrows():
                score_color = "#22c55e"
                date_str = r["DATE"].strftime("%d-%m-%Y") if pd.notna(r["DATE"]) else ""
                customer = r.get("CUSTOMER", r.get("DOMAIN", ""))
                st.markdown(
                    f'<div style="background:#1e293b;border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;border-left:3px solid {score_color}">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">'
                    f'<span style="color:#94a3b8;font-size:0.75rem">{customer}</span>'
                    f'<span style="color:{score_color};font-weight:700;font-size:0.8rem">{int(r["SCORE"])}/10</span>'
                    f'</div>'
                    f'<div style="color:#e2e8f0;font-size:0.85rem">{r["MSG_STR"]}</div>'
                    f'<div style="color:#64748b;font-size:0.7rem;margin-top:0.3rem">{date_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with col_neg:
        st.markdown("### 🔴 Negatieve Feedback")
        st.caption("Detractors (score 0-5)")
        neg = feedback[feedback["SCORE"] <= 5].head(15)
        if len(neg) == 0:
            st.info("Geen uitgebreide negatieve feedback in deze periode.")
        else:
            for _, r in neg.iterrows():
                score_color = "#ef4444"
                date_str = r["DATE"].strftime("%d-%m-%Y") if pd.notna(r["DATE"]) else ""
                customer = r.get("CUSTOMER", r.get("DOMAIN", ""))
                st.markdown(
                    f'<div style="background:#1e293b;border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;border-left:3px solid {score_color}">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">'
                    f'<span style="color:#94a3b8;font-size:0.75rem">{customer}</span>'
                    f'<span style="color:{score_color};font-weight:700;font-size:0.8rem">{int(r["SCORE"])}/10</span>'
                    f'</div>'
                    f'<div style="color:#e2e8f0;font-size:0.85rem">{r["MSG_STR"]}</div>'
                    f'<div style="color:#64748b;font-size:0.7rem;margin-top:0.3rem">{date_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Summary stats
    st.markdown("---")
    total_with_feedback = len(feedback)
    total_pos = len(feedback[feedback["SCORE"] >= 8])
    total_neg = len(feedback[feedback["SCORE"] <= 5])
    total_passive = len(feedback[(feedback["SCORE"] >= 6) & (feedback["SCORE"] <= 7)])
    st.caption(f"Totaal {total_with_feedback} uitgebreide reacties: {total_pos} positief, {total_passive} passief, {total_neg} negatief")

with tab4:
    st.subheader("Individuele Responses")

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect(
            "NPS Categorie",
            ["Promoter (8-10)", "Passive (6-7)", "Detractor (0-5)"],
            default=["Promoter (8-10)", "Passive (6-7)", "Detractor (0-5)"],
        )
    with col2:
        has_message = st.checkbox("Alleen met feedback", value=False)
    with col3:
        search_text = st.text_input("Zoek in feedback")

    resp_display = resp_filtered.copy()
    resp_display = resp_display[resp_display["NPS_CAT"].isin(cat_filter)]

    if has_message:
        resp_display = resp_display[resp_display["MESSAGE"].notna() & (resp_display["MESSAGE"] != "")]
    if search_text:
        resp_display = resp_display[
            resp_display["MESSAGE"].fillna("").str.contains(search_text, case=False)
        ]

    display_cols = ["DATE", "CUSTOMER", "DOMAIN_TYPE", "SCORE", "NPS_CAT", "MESSAGE", "PLATFORM"]
    available_cols = [c for c in display_cols if c in resp_display.columns]
    resp_show = resp_display[available_cols].sort_values("DATE", ascending=False).head(500)
    resp_show = resp_show.rename(columns={
        "DATE": "Datum", "CUSTOMER": "Klant", "DOMAIN_TYPE": "Type",
        "SCORE": "Score", "NPS_CAT": "Categorie", "MESSAGE": "Feedback",
        "PLATFORM": "Platform",
    })
    if "Feedback" in resp_show.columns:
        resp_show["Feedback"] = resp_show["Feedback"].astype(str).replace("nan", "")

    st.dataframe(resp_show.reset_index(drop=True), width="stretch", height=500)
    st.caption(f"Toont {len(resp_show)} van {len(resp_display)} responses (max 500)")
