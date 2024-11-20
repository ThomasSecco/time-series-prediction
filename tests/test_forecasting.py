import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.forecasting import forecast_arima, forecast_rf

def test_forecast_arima():
    class MockARIMA:
        def forecast(self, steps):
            return [i for i in range(steps)]

    model = MockARIMA()
    predictions = forecast_arima(model, steps=5)
    assert len(predictions) == 5
    assert predictions == [0, 1, 2, 3, 4]

def test_forecast_rf():
    model = RandomForestRegressor()
    X_test = np.array([[1], [2], [3]])
    model.fit(X_test, [100, 200, 300])
    predictions = forecast_rf(model, X_test)
    assert len(predictions) == 3
    assert predictions[0] == 100
