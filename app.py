import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from prophet import Prophet

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Budget Dashboard", page_icon="💸", layout="wide")

# ---------- THEME ----------
st.markdown("""
    <style>
    .main { background-color: #0D1117; color: #E6EDF3; }
    div[data-testid="stMetricValue"] { color: #00E6FE; font-weight: bold; }
    div.block-container{padding-top:2rem;}
    h1, h2, h3, h4, h5 { color: #00E6FE; }
    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=['csv'])
model_choice = st.sidebar.selectbox("Select Prediction Type", 
    ["Actual Spend Prediction", "Variance Prediction", "Overspending Detection", "Budget Forecasting"])

st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **AI-Powered Financial Insights**")
st.sidebar.markdown("Built with Streamlit + Scikit-learn + Prophet")

# ---------- MAIN ----------
st.title("💹 AI-Powered Budget Prediction Dashboard")
st.write("Upload your financial dataset to explore predictive insights, anomaly detection, and forecasts.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Fix Month Column if it's text like 'March' ---
    if 'Month' in df.columns:
        try:
            df['Month'] = pd.to_datetime(df['Month'], format='%B', errors='coerce')
            df['Month'] = df['Month'].fillna(
                pd.to_datetime(df['Month'], format='%b', errors='coerce')
            )
        except Exception:
            pass

    # KPI Summary
    total_spend = df['Actual_Spend'].sum()
    total_plan = df['Planned_Budget'].sum()
    variance = total_spend - total_plan

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("💰 Total Actual Spend", f"₹{total_spend:,.0f}")
    kpi2.metric("📊 Planned Budget", f"₹{total_plan:,.0f}")
    kpi3.metric("⚖️ Variance", f"₹{variance:,.0f}")

    st.markdown("---")

    # ------------------ MODEL CHOICE ------------------
    if model_choice == "Actual Spend Prediction":
        st.subheader("📈 Predict Next Month’s Actual Spend")

        if {'Planned_Budget', 'Total_Planned_Budget'}.issubset(df.columns):
            X = df[["Planned_Budget", "Total_Planned_Budget"]]
            y = df["Actual_Spend"]
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            df['Predicted_Spend'] = model.predict(X)

            # --- UPDATED LINE CHART (instead of scatter) ---
            df = df.reset_index().rename(columns={'index': 'Record'})
            fig = px.line(
                df,
                x="Record",
                y=["Actual_Spend", "Predicted_Spend"],
                labels={"value": "Spend (₹)", "Record": "Record Index"},
                title="Predicted vs Actual Spend (Line Chart)",
                template="plotly_dark"
            )
            fig.update_traces(mode="lines+markers")
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Hover to see Actual vs Predicted spend per record.")

        else:
            st.error("❌ Columns missing: Planned_Budget, Total_Planned_Budget")

    elif model_choice == "Variance Prediction":
        st.subheader("📊 Forecast Variance Based on Budget")

        if {'Planned_Budget', 'Actual_Spend'}.issubset(df.columns):
            X = df[["Planned_Budget", "Actual_Spend"]]
            y = df["Variance"]
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            df['Predicted_Variance'] = model.predict(X)

            fig = px.bar(
                df, x="Department" if "Department" in df.columns else df.index,
                y="Predicted_Variance",
                color="Category" if "Category" in df.columns else None,
                title="Predicted Variance by Department",
                hover_data=["Planned_Budget", "Actual_Spend"],
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Predicted variance shows deviation from budget plan.")
        else:
            st.error("❌ Columns missing: Planned_Budget, Actual_Spend")

    elif model_choice == "Overspending Detection":
        st.subheader("🚨 Detect Overspending Departments")

        if 'Variance' in df.columns:
            model = IsolationForest(contamination=0.15, random_state=42)
            df['Anomaly'] = model.fit_predict(df[['Variance']])
            df['Anomaly_Label'] = np.where(df['Anomaly'] == -1, '⚠️ Overspend', '✅ Normal')

            fig = px.scatter(
                df, x="Department" if "Department" in df.columns else df.index,
                y="Variance", color="Anomaly_Label",
                title="Anomaly Detection: Overspending",
                hover_data=["Category"] if "Category" in df.columns else None,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 ⚠️ indicates potential overspending anomalies.")
        else:
            st.error("❌ Column missing: Variance")

    elif model_choice == "Budget Forecasting":
        st.subheader("📅 Forecast Total Budget Trend")

        if 'Month' in df.columns and 'Actual_Spend' in df.columns:
            df_forecast = df.groupby('Month')[['Actual_Spend']].sum().reset_index()
            df_forecast.columns = ['ds', 'y']

            m = Prophet()
            m.fit(df_forecast)
            future = m.make_future_dataframe(periods=3, freq='M')
            forecast = m.predict(future)

            fig = px.line(
                forecast, x='ds', y='yhat',
                title="Predicted Spend Trend (Next 3 Months)",
                template="plotly_dark"
            )
            fig.add_scatter(x=df_forecast['ds'], y=df_forecast['y'], mode='markers', name='Actual')
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Forecast shows upcoming monthly spending trends.")
        else:
            st.error("❌ Columns missing: Month, Actual_Spend")

else:
    st.warning("⬆️ Upload your dataset to start exploring predictions.")
