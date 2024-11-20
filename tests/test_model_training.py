import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.model_training import split_data, train_rf, train_arima

def test_split_data():
    data = {
        "sales": [100, 200, 300, 400, 500],
        "feature": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)
    train, test = split_data(df)
    assert len(train) == 4  # 80% training data
    assert len(test) == 1   # 20% testing data

def test_train_rf():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = [100, 200, 300, 400]
    model = train_rf(X_train, y_train)
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "predict")

def test_train_arima():
    data = [100, 200, 300, 400, 500]
    model = train_arima(data, order=(1, 1, 0))
    assert model is not None
    assert hasattr(model, "forecast")
