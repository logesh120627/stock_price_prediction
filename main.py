
from pathlib import Path
import json
import math

import pandas as pd
import matplotlib.pyplot as plt

# Project folders
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# 1. Load stock data
data_file = DATA_DIR / "stock_data.csv"

if not data_file.exists():
    raise FileNotFoundError(
        "stock_data.csv is missing. Download the stock data first."
    )

df = pd.read_csv(data_file)

if "Close" not in df.columns:
    raise ValueError("The CSV file must contain a Close column.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df = df.dropna(subset=["Date", "Close"])

# Sort from oldest to newest
df = df.sort_values("Date").reset_index(drop=True)

if len(df) < 30:
    raise ValueError("Not enough stock data to train the model.")

# 2. Predict today's close using the previous day's close
df["Previous_Close"] = df["Close"].shift(1)
df = df.dropna().reset_index(drop=True)

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

x_train = train["Previous_Close"].tolist()
y_train = train["Close"].tolist()

x_test = test["Previous_Close"].tolist()
y_test = test["Close"].tolist()

# 3. Fit simple linear regression using Python arithmetic
x_mean = sum(x_train) / len(x_train)
y_mean = sum(y_train) / len(y_train)

denominator = sum((x - x_mean) ** 2 for x in x_train)

if denominator == 0:
    raise ValueError("The training prices do not vary.")

slope = (
    sum((x - x_mean) * (y - y_mean)
        for x, y in zip(x_train, y_train))
    / denominator
)

intercept = y_mean - slope * x_mean

# 4. Make predictions
predictions = [
    intercept + slope * x
    for x in x_test
]

# 5. Evaluate the model
errors = [
    actual - predicted
    for actual, predicted in zip(y_test, predictions)
]

mae = sum(abs(error) for error in errors) / len(errors)

rmse = math.sqrt(
    sum(error ** 2 for error in errors) / len(errors)
)

actual_mean = sum(y_test) / len(y_test)

total_variation = sum(
    (actual - actual_mean) ** 2 for actual in y_test
)

if total_variation:
    r2 = 1 - (
        sum(error ** 2 for error in errors) / total_variation
    )
else:
    r2 = float("nan")

# 6. Save the model parameters
model_info = {
    "stock_symbol": "RELIANCE.NS",
    "model": "Simple Linear Regression",
    "input_feature": "Previous_Close",
    "slope": slope,
    "intercept": intercept,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}

with open(
    MODELS_DIR / "stock_prediction_model.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(model_info, file, indent=4, allow_nan=False)

# 7. Save actual and predicted prices
results = pd.DataFrame({
    "Actual_Close": y_test,
    "Predicted_Close": predictions,
    "Previous_Close": x_test,
})

results.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

# 8. Create comparison chart
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Closing Price")
plt.plot(predictions, label="Predicted Closing Price")
plt.title("Reliance Stock Price Prediction")
plt.xlabel("Test Trading Days")
plt.ylabel("Adjusted Closing Price (INR)")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUTS_DIR / "prediction_comparison.png",
    dpi=150
)
plt.close()

# 9. Predict using the latest available closing price
latest_close = float(df["Close"].iloc[-1])
next_prediction = intercept + slope * latest_close

print("\n===== STOCK PRICE PREDICTION RESULTS =====")
print("Stock: Reliance Industries (RELIANCE.NS)")
print("Model: Simple Linear Regression")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")
print(f"MAE:  {mae:.2f} INR")
print(f"RMSE: {rmse:.2f} INR")
print(f"R²:   {r2:.4f}")
print(f"Latest closing price in dataset: {latest_close:.2f} INR")
print(f"Estimated next closing price: {next_prediction:.2f} INR")

print("\nFiles saved:")
print("- models/stock_prediction_model.json")
print("- outputs/predictions.csv")
print("- outputs/prediction_comparison.png")

print("\nNote: This is an educational model, not financial advice.")
