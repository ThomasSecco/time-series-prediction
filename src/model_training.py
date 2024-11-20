import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import joblib

def split_data(df, target_col="sales"):
    """Split data into train and test sets."""
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    return train, test

def train_arima(train_data, order=(5, 1, 0)):
    """Train an ARIMA model."""
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def train_rf(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lstm(X_train, y_train, input_shape):
    """Train an LSTM model."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model

def save_model(model, model_name):
    """Save a trained model to the models directory."""
    os.makedirs("outputs/models", exist_ok=True)
    if hasattr(model, "save"):  # For TensorFlow models
        model.save(f"outputs/models/{model_name}.h5")
    else:  # For sklearn/ARIMA models
        joblib.dump(model, f"outputs/models/{model_name}.pkl")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/preprocessed_sales.csv")
    train, test = split_data(df)
    
    # Example: Train ARIMA
    arima_model = train_arima(train["sales"])
    print(arima_model.summary())
    
    # Example: Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(
        train.drop(columns=["sales"]), train["sales"], test_size=0.2, random_state=42
    )
    rf_model = train_rf(X_train, y_train)
    
    # Example: Train LSTM
    input_shape = (X_train.shape[1], 1)
    lstm_model = train_lstm(X_train.values.reshape(-1, *input_shape), y_train, input_shape)

    # Save ARIMA model
    save_model(arima_model, "arima_sales")

    # Save Random Forest model
    save_model(rf_model, "random_forest_sales")

    # Save LSTM model
    save_model(lstm_model, "lstm_sales")
