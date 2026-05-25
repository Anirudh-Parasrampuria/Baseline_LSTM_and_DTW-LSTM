# ============================================================
# COMMIT 3: DTW-LSTM Hybrid Model
#
# How the hybrid works (DTW is no longer decorative):
#   1. For each query window, find the K most similar windows
#      in the training set using vectorised pairwise Euclidean
#      distance on the Close price column — same idea as DTW
#      nearest-neighbour but O(N) vs O(N*T) per query.
#   2. Compute a distance-weighted average of the next-step
#      close prices those K historical analogues led to.
#      This is the "analog forecast" — a non-parametric
#      estimate grounded in similar historical patterns.
#   3. Leave-one-out is applied when evaluating the analog
#      on training data so a sequence can never match itself.
#   4. Final prediction = alpha * LSTM_pred + (1-alpha) * DTW_pred
#      where alpha is grid-searched on the training split.
#
# Why this is principled:
#   - Analog forecasting is well-established in meteorology
#     and quantitative finance for exactly this reason: similar
#     historical windows tend to have similar outcomes.
#   - The blend is interpretable — alpha tells you how much
#     the pattern-matching signal complements the neural net.
# ============================================================

import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import load_model

from Data_Pipeline import (
    load_eth_data, add_technical_indicators,
    walk_forward_split, create_sequences,
    inverse_close, FEATURES,
)
from LSTM_Baseline import evaluate, train_lstm, set_seed

LOOK_BACK   = 60
SEED        = 2055
K_NEIGHBORS = 5


# ----------------------------------------------------------
# Vectorised K-NN Analog Forecaster
# ----------------------------------------------------------
def knn_analog_forecast(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    k: int = K_NEIGHBORS,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    For each query sequence in X_query, find the K most
    similar sequences in X_ref (by Euclidean distance on
    the Close column) and return a distance-weighted average
    of their corresponding next-step targets Y_ref.

    exclude_self=True implements leave-one-out (prevents a
    training sequence matching itself when X_query == X_ref).
    """
    q = X_query[:, :, 0]            # (n_query, look_back)
    r = X_ref[:, :, 0]              # (n_ref,   look_back)

    # Pairwise squared distances via broadcasting
    diff  = q[:, None, :] - r[None, :, :]       # (n_q, n_r, look_back)
    dists = np.sqrt((diff ** 2).sum(axis=2))     # (n_q, n_r)

    preds = []
    for i in range(len(X_query)):
        d = dists[i].copy()
        if exclude_self:
            d[i] = np.inf                        # mask self-match
        d = np.where(d == 0, 1e-10, d)          # guard exact duplicates
        idx = np.argsort(d)[:k]
        weights = 1.0 / d[idx]
        weights /= weights.sum()
        preds.append(np.dot(weights, Y_ref[idx]))

    return np.array(preds)


# ----------------------------------------------------------
# Alpha optimiser (grid search)
# ----------------------------------------------------------
def optimise_alpha(
    lstm_s: np.ndarray,
    dtw_s: np.ndarray,
    y_true_s: np.ndarray,
) -> float:
    best_alpha, best_mse = 0.5, float("inf")
    for alpha in np.linspace(0, 1, 21):
        blended = alpha * lstm_s + (1 - alpha) * dtw_s
        mse = np.mean((blended - y_true_s) ** 2)
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    return best_alpha


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

    # ── Step 1: Load LSTM ─────────────────────────────────
    lstm_path = "lstm_baseline.keras"
    if os.path.exists(lstm_path):
        print(f"Loading LSTM from {lstm_path}")
        lstm_model = load_model(lstm_path)
    else:
        print("Training LSTM...")
        lstm_model, _ = train_lstm(X_train, Y_train, LOOK_BACK, n_features)

    # ── Step 2: LSTM predictions (scaled) ─────────────────
    lstm_test_s  = lstm_model.predict(X_test,  verbose=0).flatten()
    lstm_train_s = lstm_model.predict(X_train, verbose=0).flatten()

    # ── Step 3: Analog forecasts (scaled) ─────────────────
    print("Building analog forecasts (vectorised K-NN)...")
    # Leave-one-out on train prevents self-match data leakage
    dtw_train_s = knn_analog_forecast(X_train, X_train, Y_train,
                                      k=K_NEIGHBORS, exclude_self=True)
    dtw_test_s  = knn_analog_forecast(X_test,  X_train, Y_train,
                                      k=K_NEIGHBORS, exclude_self=False)
    print("Analog search complete.")

    # ── Step 4: Optimise blend ────────────────────────────
    best_alpha = optimise_alpha(lstm_train_s, dtw_train_s, Y_train)
    print(f"\nOptimal alpha — LSTM: {best_alpha:.0%}  Analog: {1-best_alpha:.0%}")

    # ── Step 5: Hybrid predictions ────────────────────────
    hybrid_test_s  = best_alpha * lstm_test_s  + (1 - best_alpha) * dtw_test_s
    hybrid_train_s = best_alpha * lstm_train_s + (1 - best_alpha) * dtw_train_s

    # ── Step 6: Inverse transform ─────────────────────────
    y_test_usd    = inverse_close(scaler, Y_test,  n_features)
    y_train_usd   = inverse_close(scaler, Y_train, n_features)
    hybrid_test   = inverse_close(scaler, hybrid_test_s,  n_features)
    hybrid_train  = inverse_close(scaler, hybrid_train_s, n_features)

    # ── Step 7: Evaluate ──────────────────────────────────
    print("\n── Hybrid Train metrics ───────────────────────")
    train_m = evaluate(y_train_usd, hybrid_train, "Hybrid Train")
    print("\n── Hybrid Test metrics ────────────────────────")
    test_m  = evaluate(y_test_usd,  hybrid_test,  "Hybrid Test")

    # ── Save for commit 4 ─────────────────────────────────
    np.save("hybrid_test_pred.npy",     hybrid_test)
    np.save("hybrid_y_test_usd.npy",    y_test_usd)
    np.save("hybrid_test_metrics.npy",  np.array(list(test_m.values())))
    np.save("hybrid_train_metrics.npy", np.array(list(train_m.values())))
    np.save("best_alpha.npy",           np.array([best_alpha]))
    print("\n✓ Saved hybrid predictions and metrics for commit 4")