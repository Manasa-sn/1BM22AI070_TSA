import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load data
df = pd.read_csv("AirPassengers.csv")
df.rename(columns={"#Passengers": "Passengers"}, inplace=True)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
ts = df["Passengers"]

# 2. ACF & PACF to examine AR order
plt.figure(figsize=(10, 4))
plot_acf(ts, lags=40)
plt.title("ACF – AirPassengers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(ts, lags=40, method="ywm")
plt.title("PACF – AirPassengers")
plt.tight_layout()
plt.show()

# 3. Train–test split (last 12 as test)
train = ts[:-12]
test  = ts[-12:]

# Helper: evaluation
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# 4. AR(1) model
ar1 = AutoReg(train, lags=1).fit()
ar1_pred = ar1.predict(start=test.index[0], end=test.index[-1])
ar1_mae, ar1_mse, ar1_rmse = metrics(test, ar1_pred)

print("\nAR(1) performance:")
print(f"MAE  = {ar1_mae:.3f}")
print(f"MSE  = {ar1_mse:.3f}")
print(f"RMSE = {ar1_rmse:.3f}")

# Plot AR(1) forecast
plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(ar1_pred, label="AR(1) Forecast", linestyle="--")
plt.title("AR(1) Forecast vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Higher-lag AR(p) models
results = []
for p in [1, 2, 3, 5, 10]:
    model = AutoReg(train, lags=p).fit()
    pred = model.predict(start=test.index[0], end=test.index[-1])
    mae, mse, rmse = metrics(test, pred)
    results.append([p, mae, mse, rmse])

results_df = pd.DataFrame(results, columns=["p", "MAE", "MSE", "RMSE"])
print("\nAR(p) comparison:")
print(results_df)

# Optional: plot best AR(p)
best_p = results_df.sort_values("RMSE").iloc[0]["p"]
best_model = AutoReg(train, lags=int(best_p)).fit()
best_pred = best_model.predict(start=test.index[0], end=test.index[-1])

plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(best_pred, label=f"Best AR({int(best_p)}) Forecast", linestyle="--")
plt.title(f"Best AR Model: p = {int(best_p)}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
