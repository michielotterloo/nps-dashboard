import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
import requests
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
EXCEL_PATH = Path(__file__).parent / "data" / "NPS Dutchview.xlsx"
EXCLUDED_DOMAINS = {"dutchview.com", "mailinator.com"}


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
        resp.dropna(subset=["SCORE", "DOMAIN"])
        .groupby("DOMAIN")
        .agg(
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
date_range = st.sidebar.select_slider(
    "Periode",
    options=months_available,
    value=(months_available[0], months_available[-1]),
)

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

total_responses = len(resp_filtered)
total_scored = len(scored)
response_rate = total_scored / total_responses * 100 if total_responses > 0 else 0
avg_score = scored["SCORE"].mean() if len(scored) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("NPS Score", current_nps, delta=delta_nps, delta_label="vs vorige maand")
with col2:
    metric_card("Gem. Score", avg_score, suffix="/10")
with col3:
    metric_card("Responses", float(total_scored), suffix="")
with col4:
    metric_card("Response Rate", response_rate, suffix="%")

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Verdeling", "Klanten", "Responses"])

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
                    latest = df_plot.iloc[-1]["MAT_NPS"]
                    st.markdown(f"""
                    **MAT NPS**: {latest:.1f}
                    **Laatste maand NPS**: {df_plot.iloc[-1]['Month_NPS']:.1f}
                    **Laatste kwartaal NPS**: {df_plot.iloc[-1]['Quarter_NPS']:.1f}
                    """)
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
    min_responses = st.slider("Minimum responses", 1, 100, 10, key="min_resp")
    cust_filtered = customers[customers["Responses"] >= min_responses].copy()

    col1, col2 = st.columns(2)
    with col1:
        top_10 = cust_filtered.nlargest(15, "NPS")
        fig_top = px.bar(
            top_10, x="NPS", y="DOMAIN", orientation="h",
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
        bottom_10 = cust_filtered.nsmallest(15, "NPS")
        fig_bot = px.bar(
            bottom_10, x="NPS", y="DOMAIN", orientation="h",
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
    cust_display = cust_filtered[["DOMAIN", "Responses", "NPS", "Avg_Score", "Promoters", "Detractors"]].copy()
    cust_display["NPS"] = cust_display["NPS"].round(1)
    cust_display["Avg_Score"] = cust_display["Avg_Score"].round(2)
    cust_display = cust_display.rename(columns={
        "DOMAIN": "Klant", "Avg_Score": "Gem. Score",
    })

    search_cust = st.text_input("Zoek klant", key="search_cust")
    if search_cust:
        cust_display = cust_display[cust_display["Klant"].str.contains(search_cust, case=False, na=False)]

    st.dataframe(cust_display.reset_index(drop=True), width="stretch", height=400)

    # --- Klant detail drill-down ---
    st.markdown("---")
    st.subheader("Klant Detail")
    domain_list = cust_filtered["DOMAIN"].tolist()
    selected_domain = st.selectbox(
        "Selecteer een klant",
        options=[""] + domain_list,
        format_func=lambda x: "Kies een klant..." if x == "" else x,
        key="selected_customer",
    )

    if selected_domain:
        cust_resp = resp_filtered[resp_filtered["DOMAIN"] == selected_domain].copy()
        cust_scored = cust_resp.dropna(subset=["SCORE"])

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
                title=f"NPS Trend — {selected_domain}",
                template="plotly_dark", height=300,
                xaxis_title="Maand", yaxis_title="NPS",
                yaxis=dict(range=[-100, 100]),
                hovermode="x unified",
            )
            fig_cust.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_cust, width="stretch")

        # Responses table
        st.markdown(f"**Responses van {selected_domain}**")
        cust_show = cust_resp[["DATE", "SCORE", "NPS_CAT", "MESSAGE", "PLATFORM"]].sort_values("DATE", ascending=False)
        cust_show = cust_show.rename(columns={
            "DATE": "Datum", "SCORE": "Score", "NPS_CAT": "Categorie",
            "MESSAGE": "Feedback", "PLATFORM": "Platform",
        })
        if "Feedback" in cust_show.columns:
            cust_show["Feedback"] = cust_show["Feedback"].astype(str).replace("nan", "")
        st.dataframe(cust_show.reset_index(drop=True), width="stretch", height=400)

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

    display_cols = ["DATE", "DOMAIN", "SCORE", "NPS_CAT", "MESSAGE", "PLATFORM"]
    available_cols = [c for c in display_cols if c in resp_display.columns]
    resp_show = resp_display[available_cols].sort_values("DATE", ascending=False).head(500)
    resp_show = resp_show.rename(columns={
        "DATE": "Datum", "DOMAIN": "Klant", "SCORE": "Score",
        "NPS_CAT": "Categorie", "MESSAGE": "Feedback", "PLATFORM": "Platform",
    })
    if "Feedback" in resp_show.columns:
        resp_show["Feedback"] = resp_show["Feedback"].astype(str).replace("nan", "")

    st.dataframe(resp_show.reset_index(drop=True), width="stretch", height=500)
    st.caption(f"Toont {len(resp_show)} van {len(resp_display)} responses (max 500)")
