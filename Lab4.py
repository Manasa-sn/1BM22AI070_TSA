import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# -------- Load data --------
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
# is a pandas time-series function that forces (or converts)
# a DataFrame’s index into a specific frequency — in this case:
# "MS" → Month Start frequency
# "M" means monthly
# "S" means start of the period

# Train–test split (last 12 months as test)
train = df["Passengers"][:-12]  # Drop last 12 rows → training
test  = df["Passengers"][-12:]

# -------- Forecasting functions --------
def sma_forecast(train, test_len, window=12):
    sma = train.rolling(window).mean().dropna()
    last_val = sma.iloc[-1]
    return pd.Series([last_val] * test_len, index=test.index)

def ses_forecast(train, test_len):
    model = SimpleExpSmoothing(train).fit()
    return model.forecast(test_len)

def hw_forecast(train, test_len):
    model = ExponentialSmoothing(
        train, trend="mul", seasonal="mul", seasonal_periods=12
    ).fit()
    return model.forecast(test_len)

# -------- Metrics function --------
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# -------- Run all models --------
sma_pred = sma_forecast(train, len(test))
ses_pred = ses_forecast(train, len(test))
hw_pred  = hw_forecast(train, len(test))

rows = []
for name, pred in [("SMA", sma_pred), ("SES", ses_pred), ("Holt-Winters", hw_pred)]:
    mae, mse, rmse = get_metrics(test, pred)
    rows.append([name, mae, mse, rmse])

metrics = pd.DataFrame(rows, columns=["Model", "MAE", "MSE", "RMSE"])
print(metrics)

# -------- Plot forecasts --------
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test")
plt.plot(test.index, sma_pred, label="SMA")
plt.plot(test.index, ses_pred, label="SES")
plt.plot(test.index, hw_pred, label="Holt-Winters")
plt.legend()
plt.title("AirPassengers – SMA vs SES vs Holt-Winters")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Decomposition (trend & seasonality) --------
decomp = seasonal_decompose(df["Passengers"], model="multiplicative", period=12)
decomp.plot()
plt.tight_layout()
plt.show()
