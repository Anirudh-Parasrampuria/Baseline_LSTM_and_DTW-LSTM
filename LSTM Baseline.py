# ============================================================
# COMMIT 2: Upgraded LSTM Baseline
# Improvements over original:
#   - Dropout layers to prevent overfitting
#   - Early stopping + ReduceLROnPlateau (no more fixed epochs)
#   - Proper evaluation: RMSE, MAE, MAPE, Directional Accuracy
#   - No data leakage (uses pipeline from commit 1)
#   - Model saved to disk for comparison in commit 4
# ============================================================
 
import numpy as np
import random
import os
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF info logs
 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
from commit1_data_pipeline import (
    load_eth_data, add_technical_indicators,
    walk_forward_split, create_sequences,
    inverse_close, FEATURES,
)
 
LOOK_BACK = 60
SEED      = 2055
 
 
# ----------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
 
 
# ----------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae   = np.mean(np.abs(y_true - y_pred))
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
 
    # Directional accuracy: did we predict the move direction correctly?
    actual_dir    = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == predicted_dir) * 100
 
    tag = f"[{label}] " if label else ""
    print(f"{tag}RMSE : ${rmse:,.2f}")
    print(f"{tag}MAE  : ${mae:,.2f}")
    print(f"{tag}MAPE : {mape:.2f}%")
    print(f"{tag}Dir. Accuracy: {dir_acc:.1f}%")
    return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": dir_acc}
 
 
# ----------------------------------------------------------
# Model definition
# ----------------------------------------------------------
def build_lstm(look_back: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(look_back, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mean_squared_error")
    return model
 
 
# ----------------------------------------------------------
# Train
# ----------------------------------------------------------
def train_lstm(X_train, Y_train, look_back, n_features, epochs=100, batch_size=32):
    model = build_lstm(look_back, n_features)
 
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, verbose=1, min_lr=1e-6),
    ]
 
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,          # preserve time order within train set
    )
    return model, history
 
 
# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
 
    CSV = os.path.join(os.path.dirname(__file__), "ETH-USD.csv")
    df  = load_eth_data(CSV)
    df  = add_technical_indicators(df)
 
    train_s, test_s, scaler, split_date = walk_forward_split(df, train_ratio=0.8)
    X_train, Y_train = create_sequences(train_s, LOOK_BACK)
    X_test,  Y_test  = create_sequences(test_s,  LOOK_BACK)
 
    n_features = X_train.shape[2]
    print(f"Training LSTM baseline | {n_features} features | {len(X_train)} train seqs\n")
 
    model, history = train_lstm(X_train, Y_train, LOOK_BACK, n_features)
 
    # --- Predictions ---
    train_pred_s = model.predict(X_train, verbose=0).flatten()
    test_pred_s  = model.predict(X_test,  verbose=0).flatten()
 
    # --- Inverse transform back to USD ---
    train_pred = inverse_close(scaler, train_pred_s, n_features)
    test_pred  = inverse_close(scaler, test_pred_s,  n_features)
    y_train_usd = inverse_close(scaler, Y_train, n_features)
    y_test_usd  = inverse_close(scaler, Y_test,  n_features)
 
    # --- Evaluate ---
    print("\n── Train metrics ──────────────────────")
    train_metrics = evaluate(y_train_usd, train_pred, "Train")
    print("\n── Test metrics ───────────────────────")
    test_metrics  = evaluate(y_test_usd,  test_pred,  "Test")
 
    # --- Save model + predictions for commit 4 comparison ---
    model.save("lstm_baseline.keras")
    np.save("lstm_test_pred.npy",   test_pred)
    np.save("lstm_y_test_usd.npy",  y_test_usd)
    np.save("lstm_train_metrics.npy", np.array(list(train_metrics.values())))
    np.save("lstm_test_metrics.npy",  np.array(list(test_metrics.values())))
 
    print("\n✓ Model saved → lstm_baseline.keras")
    print("✓ Predictions saved for comparison in commit 4")