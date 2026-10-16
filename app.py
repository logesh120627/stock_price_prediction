
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction Dashboard")
st.caption("BCA Data Science Minor Project | Reliance Industries")

# Project files
DATA_FILE = Path("data/stock_data.csv")
MODEL_FILE = Path("models/stock_prediction_model.json")
PREDICTIONS_FILE = Path("outputs/predictions.csv")

# Check required files
if not DATA_FILE.exists() or not MODEL_FILE.exists():
    st.error("Required files are missing. Run `python main.py` first.")
    st.stop()

# Load files
df = pd.read_csv(DATA_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)

with MODEL_FILE.open("r", encoding="utf-8") as file:
    model = json.load(file)

# Show metrics
st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"₹{model['MAE']:.2f}")
c2.metric("RMSE", f"₹{model['RMSE']:.2f}")
c3.metric("R² Score", f"{model['R2']:.4f}")

# Latest data and prediction
st.subheader("Latest Stock Price")


# Ensure dates and prices are valid and sorted
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df = (
    df.dropna(subset=["Date", "Close"])
      .sort_values("Date")
      .reset_index(drop=True)
)

# Latest available observation in the dataset
latest_row = df.iloc[-1]
latest_close = float(latest_row["Close"])
latest_date = latest_row["Date"].strftime("%d %B %Y")

# Forecast from the saved regression model
estimated_price = (
    model["intercept"] + model["slope"] * latest_close
)

st.subheader("Latest Available Stock Data")

c1, c2, c3 = st.columns(3)
c1.metric("Latest Closing Price", f"₹{latest_close:.2f}")
c2.metric("Data Through", latest_date)
c3.metric("Model Estimate", f"₹{estimated_price:.2f}")

st.warning(
    "This dataset is historical and ends on the date shown above. "
    "The estimate is not a live market quote or a guaranteed forecast."
)

estimated_price = (
    model["intercept"] + model["slope"] * latest_close
)

c1, c2 = st.columns(2)
c1.metric("Latest Price in Dataset", f"₹{latest_close:.2f}")
c2.metric("Estimated Next Price", f"₹{estimated_price:.2f}")

st.caption(
    "The estimate is generated from the saved regression model. "
    "It is an educational forecast, not a guaranteed future price."
)

# Historical price chart
st.subheader("Historical Closing Prices")

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

fig, ax = plt.subplots(figsize=(12, 5))
if "Date" in df.columns:
    ax.plot(df["Date"], df["Close"])
else:
    ax.plot(df["Close"])

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (INR)")
ax.set_title("Reliance Industries Historical Closing Prices")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Actual vs predicted
st.subheader("Actual vs Predicted Prices")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(predictions["Actual_Close"], label="Actual")
ax2.plot(predictions["Predicted_Close"], label="Predicted")
ax2.set_xlabel("Test Trading Days")
ax2.set_ylabel("Closing Price (INR)")
ax2.legend()
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# Prediction records
with st.expander("View prediction records"):
    st.dataframe(predictions, width="stretch")

st.success("Dashboard loaded successfully.")
