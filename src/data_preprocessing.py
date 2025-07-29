import pandas as pd
import os

def load_and_preprocess_data(csv_path):
    """
    Load the raw Bitcoin price data and preprocess it to daily data.
    Args:
        csv_path (str): Path to the raw CSV file.
    Returns:
        pd.DataFrame: Preprocessed daily data.
    """
    bitcoin_data = pd.read_csv(csv_path)
    # Resample to daily data (assuming 1440 minutes per day)
    df = bitcoin_data[::1440].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def select_features(df):
    X = df[['Timestamp', 'Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    return X, y

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '../data/btcusd_1-min_data.csv')
    df = load_and_preprocess_data(data_path)
    print(df.head()) 