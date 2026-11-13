import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="B2B Sales Intelligence",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    sales = pd.read_parquet(BASE_DIR / "cleaned_data.parquet")
    customers = pd.read_parquet(BASE_DIR / "customer_segments.parquet")
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    return sales, customers

df, rfm = load_data()

st.title("B2B Sales Intelligence")
st.caption("Sales, Marketing & Supply Chain Decision Support | 2020")

# SIDEBAR FILTERS
st.sidebar.header("Dashboard Filters")

date_range = st.sidebar.date_input(
    "Invoice Date",
    value=(df["InvoiceDate"].min().date(),
           df["InvoiceDate"].max().date()),
    min_value=df["InvoiceDate"].min().date(),
    max_value=df["InvoiceDate"].max().date()
)

categories = st.sidebar.multiselect(
    "Product Categories",
    sorted(df["Category"].dropna().unique().tolist())
)

cities = st.sidebar.multiselect(
    "Cities",
    sorted(df["City"].dropna().unique().tolist())
)

filtered = df.copy()

if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        filtered["InvoiceDate"].dt.date.between(start, end)
    ]

if categories:
    filtered = filtered[filtered["Category"].isin(categories)]

if cities:
    filtered = filtered[filtered["City"].isin(cities)]

# KPI CALCULATIONS
revenue = filtered["Revenue"].sum()
orders = filtered["InvoiceNo"].nunique()
customers = filtered["CustomerID"].nunique()
units = filtered["Quantity"].sum()
aov = revenue / orders if orders else 0

tab1, tab2, tab3, tab4 = st.tabs([
    "Sales Overview",
    "Product Analytics",
    "Customer Intelligence",
    "Supply Chain"
])

with tab1:
    st.subheader("Executive Sales Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Net Revenue", f"{revenue:,.0f}")
    c2.metric("Orders", f"{orders:,}")
    c3.metric("Customers", f"{customers:,}")
    c4.metric("Units Sold", f"{units:,.0f}")
    c5.metric("Average Order Value", f"{aov:,.2f}")

    monthly = (
        filtered.groupby("YearMonth", as_index=False)
        .agg(Revenue=("Revenue", "sum"),
             Orders=("InvoiceNo", "nunique"))
        .sort_values("YearMonth")
    )

    fig = px.line(
        monthly,
        x="YearMonth",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

    if not monthly.empty:
        best = monthly.loc[monthly["Revenue"].idxmax()]
        worst = monthly.loc[monthly["Revenue"].idxmin()]

        st.info(
            f"Highest revenue: {best['YearMonth']} | "
            f"Lowest revenue: {worst['YearMonth']}"
        )

with tab2:
    st.subheader("Product and Category Performance")

    category = (
        filtered.groupby("Category", as_index=False)
        .agg(Revenue=("Revenue", "sum"),
             Units=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(15)
    )

    fig = px.bar(
        category.sort_values("Revenue"),
        x="Revenue",
        y="Category",
        orientation="h",
        title="Top 15 Categories by Revenue",
        color="Revenue",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    products = (
        filtered.groupby(["StockCode", "Description"], as_index=False)
        .agg(Revenue=("Revenue", "sum"),
             Units=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(20)
    )

    st.subheader("Top 20 Products")
    st.dataframe(products, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("RFM Customer Segmentation")

    # Show segment distribution for customers
    # who appear in the current filtered data.
    selected_ids = filtered["CustomerID"].unique()
    current_rfm = rfm[rfm["CustomerID"].isin(selected_ids)]

    seg = (
        current_rfm.groupby("Segment", as_index=False)
        .agg(Customers=("CustomerID", "count"),
             RFM_Monetary=("Monetary", "sum"))
    )

    fig = px.bar(
        seg,
        x="Segment",
        y="Customers",
        color="Segment",
        text="Customers",
        title="Customer Segments"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "RFM scores are based on the full 2020 dataset. "
        "Segment monetary values are historical totals."
    )

    st.dataframe(
        seg,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Marketing Recommendations")
    st.write("**Champions:** Prioritize retention and VIP benefits.")
    st.write("**Loyal Customers:** Develop cross-selling campaigns.")
    st.write("**Potential Customers:** Encourage repeat purchases.")
    st.write("**At Risk:** Test targeted reactivation campaigns.")

with tab4:
    st.subheader("Supply Chain and Demand Insights")

    city_demand = (
        filtered.groupby("City", as_index=False)
        .agg(Units=("Quantity", "sum"),
             Revenue=("Revenue", "sum"))
        .sort_values("Units", ascending=False)
        .head(15)
    )

    fig = px.bar(
        city_demand,
        x="City",
        y="Units",
        title="Top 15 Cities by Product Demand"
    )
    st.plotly_chart(fig, use_container_width=True)

    demand = (
        filtered.groupby("YearMonth", as_index=False)
        ["Quantity"].sum()
        .sort_values("YearMonth")
    )

    demand["RollingAverage3M"] = (
        demand["Quantity"].rolling(3).mean()
    )

    fig = px.line(
        demand,
        x="YearMonth",
        y=["Quantity", "RollingAverage3M"],
        markers=True,
        title="Monthly Demand and 3-Month Moving Average"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Demand trends can support inventory planning. "
        "Actual stock levels and lead times are not available."
    )

st.divider()
st.caption(
    "Source: B2B Transaction Dataset (2020). "
    "Revenue assumes NetPrice is a discounted unit price."
)


# FORECASTING SECTION
st.divider()
st.header("Sales Forecasting")

forecast_df = pd.read_parquet(
    BASE_DIR / "sales_forecast.parquet"
)

historical = (
    df.groupby("YearMonth", as_index=False)
    ["Revenue"].sum()
    .sort_values("YearMonth")
)

historical["Date"] = pd.to_datetime(
    historical["YearMonth"]
)

import plotly.graph_objects as go

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=historical["Date"],
    y=historical["Revenue"],
    mode="lines+markers",
    name="Historical Revenue"
))

fig_forecast.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["ForecastRevenue"],
    mode="lines+markers",
    name="Forecast",
    line=dict(dash="dash")
))

fig_forecast.update_layout(
    title="3-Month Revenue Forecast",
    xaxis_title="Month",
    yaxis_title="Revenue",
    template="plotly_dark"
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Forecasting method: Recursive 3-month moving average. "
    "Predictions are illustrative and based only on 2020 sales."
)
