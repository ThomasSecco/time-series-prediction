import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

def preprocess_data(df, target_col="sales"):
    """Preprocess the dataset."""
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    
    # Feature engineering: Extract date parts
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Scale the target variable
    scaler = MinMaxScaler()
    df[target_col] = scaler.fit_transform(df[[target_col]])
    return df, scaler

if __name__ == "__main__":
    file_path = "data/raw/store_sales.csv"
    df = load_data(file_path)
    df, scaler = preprocess_data(df)
    df.to_csv("data/processed/preprocessed_sales.csv", index=False)
