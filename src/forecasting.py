import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def forecast_arima(model, steps=30):
    """Forecast using ARIMA model."""
    forecast = model.forecast(steps=steps)
    return forecast

def forecast_rf(model, X_test):
    """Forecast using Random Forest."""
    predictions = model.predict(X_test)
    return predictions

def forecast_lstm(model, X_test, input_shape):
    """Forecast using LSTM."""
    predictions = model.predict(X_test.reshape(-1, *input_shape))
    return predictions.flatten()

def plot_forecast(actual, forecast, title="Forecast vs Actual"):
    """Plot forecast vs actual data."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual")
    plt.plot(forecast, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.show()
