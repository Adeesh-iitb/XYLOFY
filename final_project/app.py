"""
Superstore Sales Forecasting & Demand Intelligence Dashboard
Task 7 deliverable — deploy this on Streamlit Community Cloud.

To run locally:
    pip install streamlit pandas numpy matplotlib seaborn statsmodels scikit-learn plotly
    streamlit run app.py

To deploy on Streamlit Community Cloud:
    1. Push this folder (app.py, requirements.txt, data/train.csv, data/vgsales.csv) to a public GitHub repo.
    2. Go to https://share.streamlit.io, sign in, "New app", point it at the repo/app.py.
    3. Submit the resulting *.streamlit.app URL as your live link.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")


# --------------------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


@st.cache_data
def get_monthly_series(_df, category=None, region=None):
    d = _df
    if category:
        d = d[d["Category"] == category]
    if region:
        d = d[d["Region"] == region]
    ts = d.set_index("Order Date").resample("MS")["Sales"].sum()
    ts.index.freq = "MS"
    return ts


@st.cache_data
def get_weekly_series(_df):
    ts = _df.set_index("Order Date").resample("W")["Sales"].sum()
    return ts


@st.cache_resource
def fit_sarima_and_forecast(ts, horizon):
    train = ts.iloc[:-horizon] if len(ts) > horizon + 6 else ts
    test = ts.iloc[-horizon:] if len(ts) > horizon + 6 else None
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc_obj = fit.get_forecast(steps=horizon)
    pred = fc_obj.predicted_mean
    ci = fc_obj.conf_int()

    mae = rmse = None
    if test is not None:
        mae = np.mean(np.abs(test.values - pred.values[:len(test)]))
        rmse = np.sqrt(np.mean((test.values - pred.values[:len(test)]) ** 2))
    return pred, ci, mae, rmse


df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# --------------------------------------------------------------------------------------
# PAGE 1 — Sales Overview Dashboard
# --------------------------------------------------------------------------------------
if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    yearly = df.groupby("Year")["Sales"].sum().reset_index()
    fig1 = px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year", text_auto=".2s")
    st.plotly_chart(fig1, use_container_width=True)

    monthly = df.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    fig2 = px.line(monthly, x="Order Date", y="Sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region and Category")
    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Filter Region(s)", sorted(df["Region"].unique()),
                                        default=sorted(df["Region"].unique()))
    with col2:
        category_filter = st.multiselect("Filter Categor(y/ies)", sorted(df["Category"].unique()),
                                          default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig3 = px.bar(grouped, x="Region", y="Sales", color="Category", barmode="group",
                  title="Sales by Region and Category (filtered)")
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# --------------------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    dim_type = st.selectbox("Forecast by", ["Category", "Region"])
    if dim_type == "Category":
        dim_value = st.selectbox("Select Category", sorted(df["Category"].unique()))
        ts = get_monthly_series(df, category=dim_value)
    else:
        dim_value = st.selectbox("Select Region", sorted(df["Region"].unique()))
        ts = get_monthly_series(df, region=dim_value)

    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    with st.spinner("Fitting SARIMA model..."):
        pred, ci, mae, rmse = fit_sarima_and_forecast(ts, horizon)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts.index[-18:], ts.values[-18:], label="Actual", marker="o")
    ax.plot(pred.index, pred.values, label="Forecast", marker="o", color="red")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.15, label="95% CI")
    ax.set_title(f"{horizon}-Month Forecast: {dim_value} ({dim_type})")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecasted values")
    st.dataframe(pred.rename("Forecast").to_frame().style.format("{:,.0f}"))

    if mae is not None:
        c1, c2 = st.columns(2)
        c1.metric("MAE (last held-out months)", f"{mae:,.0f}")
        c2.metric("RMSE (last held-out months)", f"{rmse:,.0f}")
    else:
        st.info("Not enough history in this segment to compute a held-out MAE/RMSE.")

# --------------------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# --------------------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")

    weekly_ts = get_weekly_series(df)
    iso = IsolationForest(contamination=0.07, random_state=42)
    labels = iso.fit_predict(weekly_ts.values.reshape(-1, 1))
    anomalies = weekly_ts[labels == -1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_ts.index, weekly_ts.values, label="Weekly Sales", color="steelblue")
    ax.scatter(anomalies.index, anomalies.values, color="red", s=60, zorder=5, label="Anomaly")
    ax.legend()
    ax.set_title("Weekly Sales with Detected Anomalies (Isolation Forest)")
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    anomaly_table = anomalies.rename("Sales").reset_index().rename(columns={"Order Date": "Week Starting"})
    st.dataframe(anomaly_table.style.format({"Sales": "{:,.0f}"}))

# --------------------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# --------------------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.title("🧩 Product Demand Segments")

    sub_cat = df.copy()
    sub_cat["YearMonth"] = sub_cat["Order Date"].dt.to_period("M")
    total_sales = sub_cat.groupby("Sub-Category")["Sales"].sum()
    yearly_sc = sub_cat.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
    piv = yearly_sc.pivot(index="Sub-Category", columns="Year", values="Sales").fillna(0)
    growth_rate = ((piv[piv.columns[-1]] - piv[piv.columns[0]]) / piv[piv.columns[0]].replace(0, np.nan)) * 100
    monthly_sc = sub_cat.groupby(["Sub-Category", "YearMonth"])["Sales"].sum().reset_index()
    volatility = monthly_sc.groupby("Sub-Category")["Sales"].std()
    order_val = sub_cat.groupby("Sub-Category")["Sales"].mean()

    features_df = pd.DataFrame({
        "total_sales": total_sales, "growth_rate_pct": growth_rate,
        "volatility": volatility, "avg_order_value": order_val,
    }).dropna()

    X_scaled = StandardScaler().fit_transform(features_df)
    k = st.slider("Number of clusters (k)", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features_df["cluster"] = kmeans.fit_predict(X_scaled)

    pcs = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    features_df["pc1"], features_df["pc2"] = pcs[:, 0], pcs[:, 1]

    fig = px.scatter(features_df.reset_index(), x="pc1", y="pc2", color=features_df["cluster"].astype(str),
                      text="Sub-Category", title="Product Sub-Category Demand Clusters (PCA projection)")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-categories by cluster")
    st.dataframe(features_df[["cluster", "total_sales", "growth_rate_pct", "volatility", "avg_order_value"]]
                 .style.format({"total_sales": "{:,.0f}", "growth_rate_pct": "{:.1f}",
                                 "volatility": "{:,.0f}", "avg_order_value": "{:,.0f}"}))
