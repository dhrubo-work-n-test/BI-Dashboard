import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="ML Forecasting Dashboard", page_icon="📊", layout="wide")

# --- TITLE ---
st.title("🤖 ML-Powered Budget & Spend Dashboard")
st.markdown("### Explore spend predictions, variances, and insights")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data successfully loaded!")
    st.dataframe(df.head())
else:
    st.info("👆 Please upload your CSV file to continue.")
    st.stop()

# --- BASIC INFO ---
st.subheader("📋 Data Overview")
st.write(f"Shape of dataset: {df.shape}")
st.write("Columns:", list(df.columns))

# --- SIMPLE REGRESSION MODEL ---
# Using Planned_Budget -> Actual_Spend for predictions
if "Planned_Budget" in df.columns and "Actual_Spend" in df.columns:
    X = df[["Planned_Budget"]]
    y = df["Actual_Spend"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    df["Predicted_Spend"] = model.predict(X)

    st.subheader("📈 Predicted vs Actual Spend (Linear Chart)")

    # --- LINE CHART: Actual vs Predicted Spend ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Actual_Spend'],
        mode='lines+markers',
        name='Actual Spend',
        line=dict(color='cyan', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Predicted_Spend'],
        mode='lines+markers',
        name='Predicted Spend',
        line=dict(color='magenta', width=2, dash='dash')
    ))

    fig.update_layout(
        title='📊 Predicted vs Actual Spend',
        xaxis_title='Record Index / Month',
        yaxis_title='Spend (₹)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            title='Legend',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Hover Purpose Note ---
    with st.expander("ℹ️ Chart Purpose"):
        st.write("""
        **Purpose:**  
        This chart compares the predicted spend (using a regression model) against the actual spend values.  
        It helps identify under/overspending patterns and evaluate model performance visually.
        """)
else:
    st.error("❌ Required columns 'Planned_Budget' and 'Actual_Spend' not found in dataset.")
