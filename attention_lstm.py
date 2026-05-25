# ============================================================
# COMMIT 5: LSTM + Self-Attention
#
# Architecture change:
#   Input (60, 13)
#     → LSTM(128, return_sequences=True)   # encode each timestep
#     → Dropout(0.2)
#     → LSTM(64, return_sequences=True)    # keep all 60 hidden states
#     → Dropout(0.2)
#     → Temporal Self-Attention            # learn which days matter
#     → context vector (64,)              # weighted sum of hidden states
#     → Dense(32, relu) → Dense(1)
#
# Why attention helps here:
#   A standard LSTM collapses the 60-day window into a single
#   hidden state, giving equal "memory" weight to day 1 and
#   day 59. Attention lets the model learn a soft weighting
#   over all 60 timesteps so it can focus on, e.g., the last
#   5 days before a breakout while suppressing a quiet period
#   three weeks earlier.
#
#   Crucially, we can EXTRACT and VISUALIZE those weights —
#   making the model interpretable, not just accurate.
# ============================================================

import numpy as np
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Layer, Permute, Multiply, Lambda,
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Data_Pipeline import (
    load_eth_data, add_technical_indicators,
    walk_forward_split, create_sequences,
    inverse_close, FEATURES,
)
from LSTM_Baseline import evaluate, set_seed

LOOK_BACK  = 60
SEED       = 2055


# ----------------------------------------------------------
# Temporal Attention Layer
# ----------------------------------------------------------
class TemporalAttention(Layer):
    """
    Additive (Bahdanau-style) attention over the time axis.
    Given LSTM hidden states h of shape (batch, T, units):
      1. score_t = tanh(W·h_t + b)          energy per timestep
      2. alpha_t = softmax(score_t)          normalised attention weights
      3. context = Σ alpha_t * h_t           weighted context vector
    Returns (context, alpha) so weights can be extracted later.
    """
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, use_bias=True,  activation="tanh", name="attn_W")
        self.v = Dense(1,     use_bias=False, name="attn_v")

    def call(self, hidden_states):
        # hidden_states: (batch, T, units)
        score  = self.v(self.W(hidden_states))          # (batch, T, 1)
        alpha  = tf.nn.softmax(score, axis=1)           # (batch, T, 1)
        context = tf.reduce_sum(alpha * hidden_states, axis=1)  # (batch, units)
        return context, alpha

    def get_config(self):
        cfg = super().get_config()
        return cfg


# ----------------------------------------------------------
# Model builder
# ----------------------------------------------------------
def build_attention_lstm(look_back: int, n_features: int, lstm_units: int = 64):
    """
    Returns the full model and a separate extractor model that
    outputs (prediction, attention_weights) for interpretability.
    """
    inp = Input(shape=(look_back, n_features), name="input")

    x = LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x = Dropout(0.2, name="drop_1")(x)
    x = LSTM(lstm_units, return_sequences=True, name="lstm_2")(x)
    x = Dropout(0.2, name="drop_2")(x)

    attn_layer = TemporalAttention(lstm_units, name="attention")
    context, alpha = attn_layer(x)          # (batch, units), (batch, T, 1)

    out = Dense(32, activation="relu", name="dense_1")(context)
    out = Dense(1, name="output")(out)

    # Training model
    model = Model(inputs=inp, outputs=out, name="LSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
    )

    # Extractor model — outputs predictions AND attention weights
    extractor = Model(
        inputs=inp,
        outputs=[out, alpha],
        name="LSTM_Attention_Extractor",
    )

    return model, extractor


# ----------------------------------------------------------
# Train
# ----------------------------------------------------------
def train_attention_lstm(X_train, Y_train, look_back, n_features, epochs=100, batch_size=32):
    model, extractor = build_attention_lstm(look_back, n_features)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=6, verbose=1, min_lr=1e-6),
    ]

    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )
    return model, extractor


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

    print(f"Training Attention-LSTM | {n_features} features | {len(X_train)} train seqs\n")
    model, extractor = train_attention_lstm(X_train, Y_train, LOOK_BACK, n_features)

    # ── Predictions ───────────────────────────────────────
    test_pred_s, test_attn  = extractor.predict(X_test,  verbose=0)
    train_pred_s, train_attn = extractor.predict(X_train, verbose=0)

    test_pred  = inverse_close(scaler, test_pred_s.flatten(),  n_features)
    train_pred = inverse_close(scaler, train_pred_s.flatten(), n_features)
    y_test_usd  = inverse_close(scaler, Y_test,  n_features)
    y_train_usd = inverse_close(scaler, Y_train, n_features)

    # ── Evaluate ──────────────────────────────────────────
    print("\n── Attention-LSTM Train metrics ───────────────")
    train_m = evaluate(y_train_usd, train_pred, "Attn Train")
    print("\n── Attention-LSTM Test metrics ────────────────")
    test_m  = evaluate(y_test_usd,  test_pred,  "Attn Test")

    # ── Save everything ───────────────────────────────────
    model.save("attention_lstm.keras")

    # Attention weights: shape (n_samples, T, 1) → (n_samples, T)
    np.save("attn_test_weights.npy",   test_attn.squeeze(-1))
    np.save("attn_train_weights.npy",  train_attn.squeeze(-1))
    np.save("attn_test_pred.npy",      test_pred)
    np.save("attn_y_test_usd.npy",     y_test_usd)
    np.save("attn_test_metrics.npy",   np.array(list(test_m.values())))
    np.save("attn_train_metrics.npy",  np.array(list(train_m.values())))

    # ── Print attention summary ────────────────────────────
    avg_attn = test_attn.squeeze(-1).mean(axis=0)   # (T,)
    top5_days = np.argsort(avg_attn)[-5:][::-1]
    print("\nTop 5 most-attended days in the 60-day window (averaged over test set):")
    for rank, day in enumerate(top5_days, 1):
        days_ago = LOOK_BACK - day
        print(f"  #{rank}  Day {day:>2}  ({days_ago:>2} days before prediction) — weight: {avg_attn[day]:.4f}")

    print("\n✓ Model + attention weights saved.")
    print("✓ Run commit6_attention_viz.py to generate heatmap visualisation.")
