# ============================================================
# COMMIT 1: Data Pipeline
# - Replaces raw Close + naive MAs with proper technical indicators
# - Fixes data leakage: scaler is fit ONLY on training data
# - Adds walk-forward split utility used by all downstream models
# - Adds log-return feature (stationary, better for neural nets)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------------------
# 1. Load & basic clean
# ----------------------------------------------------------
def load_eth_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_index()
    df = df[df["Close"] > 0]          # drop any zero/bad rows
    return df


# ----------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    v = df["Volume"]

    # Trend
    df["MA_5"]  = c.rolling(5).mean()
    df["MA_20"] = c.rolling(20).mean()
    df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = c.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (14-period)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period)
    rolling_std = c.rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * rolling_std
    df["BB_lower"] = df["MA_20"] - 2 * rolling_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (df["MA_20"] + 1e-9)

    # Volume momentum
    df["Volume_MA_5"] = v.rolling(5).mean()
    df["Volume_ratio"] = v / (df["Volume_MA_5"] + 1e-9)

    # Log return (stationary price representation)
    df["Log_return"] = np.log(c / c.shift(1))

    # Price momentum
    df["Momentum_5"]  = c.pct_change(5)
    df["Momentum_10"] = c.pct_change(10)

    df = df.dropna()
    return df


# ----------------------------------------------------------
# 3. Feature list (Close must stay first — it's the target)
# ----------------------------------------------------------
FEATURES = [
    "Close",
    "MA_5", "MA_20", "EMA_12", "EMA_26",
    "MACD", "MACD_signal",
    "RSI",
    "BB_width",
    "Volume_ratio",
    "Log_return",
    "Momentum_5", "Momentum_10",
]


# ----------------------------------------------------------
# 4. Walk-forward split  (no leakage)
# ----------------------------------------------------------
def walk_forward_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Returns train_df, test_df.
    Scaler is fit on train only and returned so test can be transformed
    with the same parameters — preventing look-ahead bias.
    """
    data = df[FEATURES].values
    split = int(len(data) * train_ratio)

    train_raw = data[:split]
    test_raw  = data[split:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)   # fit on train ONLY
    test_scaled  = scaler.transform(test_raw)        # transform test with same params

    return train_scaled, test_scaled, scaler, df.index[split]


# ----------------------------------------------------------
# 5. Sequence builder
# ----------------------------------------------------------
def create_sequences(scaled_data: np.ndarray, look_back: int = 60):
    """
    Returns X of shape (n_samples, look_back, n_features)
    and Y of shape (n_samples,) — the next close price (column 0).
    """
    X, Y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        X.append(scaled_data[i : i + look_back, :])
        Y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(Y)


# ----------------------------------------------------------
# 6. Inverse transform helper (only Close column)
# ----------------------------------------------------------
def inverse_close(scaler: MinMaxScaler, values: np.ndarray, n_features: int) -> np.ndarray:
    """
    Inverse-transform a 1-D array of scaled Close values back to USD.
    Pads with zeros for the other feature columns.
    """
    padded = np.hstack([values.reshape(-1, 1),
                        np.zeros((len(values), n_features - 1))])
    return scaler.inverse_transform(padded)[:, 0]


# ----------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------
if __name__ == "__main__":
    import os
    CSV = os.path.join(os.path.dirname(__file__), "ETH-USD.csv")

    df = load_eth_data(CSV)
    df = add_technical_indicators(df)

    print(f"Dataset shape after feature engineering: {df[FEATURES].shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Features ({len(FEATURES)}):\n  {FEATURES}")
    print(f"\nSample (last 3 rows):\n{df[FEATURES].tail(3).round(4)}")

    train_s, test_s, scaler, split_date = walk_forward_split(df)
    print(f"\nTrain rows: {len(train_s)}  |  Test rows: {len(test_s)}")
    print(f"Train/test split date: {split_date.date()}")

    X_tr, Y_tr = create_sequences(train_s, look_back=60)
    X_te, Y_te = create_sequences(test_s,  look_back=60)
    print(f"\nX_train: {X_tr.shape}  Y_train: {Y_tr.shape}")
    print(f"X_test : {X_te.shape}  Y_test : {Y_te.shape}")
    print("\n✓ Data pipeline OK")
