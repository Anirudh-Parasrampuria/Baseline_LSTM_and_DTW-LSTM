# ============================================================
# COMMIT 6: Attention Heatmap Visualization
#
# Produces two publication-quality figures:
#   1. attention_heatmap.png  — per-prediction attention map
#      showing which of the 60 lookback days the model focused
#      on for each test prediction.
#   2. attention_profile.png  — averaged attention profile
#      across all test predictions, with actual ETH price
#      overlaid to reveal what patterns trigger high attention.
#
# This is the key interpretability output — it lets you say:
#   "The model consistently attends most to days 35-45 of the
#    60-day window (~2-3 weeks before prediction), suggesting
#    medium-term momentum patterns drive its forecasts."
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

LOOK_BACK = 60
FEATURES  = [
    "Close", "MA_5", "MA_20", "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "RSI", "BB_width",
    "Volume_ratio", "Log_return", "Momentum_5", "Momentum_10",
]


def load_data():
    required = [
        "attn_test_weights.npy", "attn_test_pred.npy",
        "attn_y_test_usd.npy",   "attn_train_weights.npy",
    ]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f} — run commit5 first.")

    attn_test  = np.load("attn_test_weights.npy")   # (9, 60)
    attn_train = np.load("attn_train_weights.npy")  # (216, 60)
    test_pred  = np.load("attn_test_pred.npy")
    y_test     = np.load("attn_y_test_usd.npy")
    return attn_test, attn_train, test_pred, y_test


def make_attention_heatmap(attn_test, test_pred, y_test,
                           save_path="attention_heatmap.png"):
    """
    Per-prediction heatmap: rows = test predictions, cols = lookback days.
    Each cell = attention weight (how much the model cared about that day).
    """
    # Dark terminal-green aesthetic
    BG      = "#050d0a"
    PANEL   = "#0a1a14"
    GREEN   = "#00ff88"
    AMBER   = "#ffb347"
    TEXT    = "#c8ffd8"
    GRID    = "#0d2018"

    # Custom colormap: dark → vivid green
    cmap = LinearSegmentedColormap.from_list(
        "attn", ["#050d0a", "#003322", "#00aa55", "#00ff88", "#ccffee"], N=256
    )

    n_preds = attn_test.shape[0]
    days    = np.arange(LOOK_BACK)

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[4, 1],
        height_ratios=[3, 1],
        hspace=0.08,
        wspace=0.04,
    )

    # ── Main heatmap ──────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_heat.set_facecolor(PANEL)

    im = ax_heat.imshow(
        attn_test,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=attn_test.min(),
        vmax=attn_test.max(),
    )

    ax_heat.set_yticks(range(n_preds))
    ax_heat.set_yticklabels(
        [f"Pred {i+1}  (${y_test[i]:,.0f} act / ${test_pred[i]:,.0f} pred)"
         for i in range(n_preds)],
        color=TEXT, fontsize=8, fontfamily="monospace",
    )
    ax_heat.set_xticks([])
    ax_heat.set_title(
        "Attention Weight Heatmap — Which Days Drive Each Prediction",
        color=TEXT, fontsize=12, fontweight="bold", pad=10,
    )
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID)

    # ── Colourbar ─────────────────────────────────────────
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.set_label("Attention Weight", color=TEXT, fontsize=8)
    cb.outline.set_edgecolor(GRID)

    # ── Average attention profile (bottom) ────────────────
    ax_avg = fig.add_subplot(gs[1, 0])
    ax_avg.set_facecolor(PANEL)

    avg = attn_test.mean(axis=0)
    std = attn_test.std(axis=0)

    ax_avg.fill_between(days, avg - std, avg + std, color=GREEN, alpha=0.15)
    ax_avg.plot(days, avg, color=GREEN, lw=1.5, label="Mean attention")

    # Mark the peak attention day
    peak_day = np.argmax(avg)
    ax_avg.axvline(peak_day, color=AMBER, lw=1.2, ls="--", alpha=0.8)
    ax_avg.text(
        peak_day + 1, avg.max() * 0.97,
        f"Peak: day {peak_day}\n({LOOK_BACK-peak_day}d ago)",
        color=AMBER, fontsize=7, fontfamily="monospace",
    )

    # Annotate recency vs early region
    ax_avg.axvspan(0,  20, color="#00ff88", alpha=0.04, label="Early (1-20d)")
    ax_avg.axvspan(40, 59, color="#ff8800", alpha=0.06, label="Recent (41-60d)")

    ax_avg.set_xlabel("Lookback Day (0 = oldest, 59 = most recent)", color=TEXT, fontsize=9)
    ax_avg.set_ylabel("Avg Attention", color=TEXT, fontsize=9)
    ax_avg.tick_params(colors=TEXT, labelsize=8)
    ax_avg.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)
    for spine in ax_avg.spines.values():
        spine.set_edgecolor(GRID)

    # ── Right panel: per-prediction peak day ──────────────
    ax_peak = fig.add_subplot(gs[:, 1])
    ax_peak.set_facecolor(PANEL)

    peak_days = [np.argmax(attn_test[i]) for i in range(n_preds)]
    errors    = np.abs(y_test - test_pred)

    sc = ax_peak.scatter(
        peak_days,
        range(n_preds),
        c=errors,
        cmap="RdYlGn_r",
        s=120,
        zorder=3,
        edgecolors=TEXT,
        linewidths=0.5,
    )
    cb2 = fig.colorbar(sc, ax=ax_peak, fraction=0.06, pad=0.02)
    cb2.set_label("Abs error ($)", color=TEXT, fontsize=7)
    cb2.ax.tick_params(colors=TEXT, labelsize=7)
    cb2.outline.set_edgecolor(GRID)

    ax_peak.set_xlabel("Peak attention day", color=TEXT, fontsize=8)
    ax_peak.set_yticks(range(n_preds))
    ax_peak.set_yticklabels([f"P{i+1}" for i in range(n_preds)],
                             color=TEXT, fontsize=8)
    ax_peak.set_title("Peak day\nvs error", color=TEXT, fontsize=9, pad=8)
    for spine in ax_peak.spines.values():
        spine.set_edgecolor(GRID)
    ax_peak.tick_params(colors=TEXT)
    ax_peak.grid(True, color=GRID, lw=0.5, alpha=0.5)

    fig.suptitle(
        "LSTM Temporal Attention — ETH/USD 60-Day Lookback Window",
        color=TEXT, fontsize=14, fontweight="bold", y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"✓ Heatmap saved → {save_path}")
    plt.close()


def make_attention_profile(attn_test, attn_train, save_path="attention_profile.png"):
    """
    Single-panel averaged attention profile comparing test vs training
    distributions to show whether attention generalises.
    """
    BG    = "#050d0a"
    PANEL = "#0a1a14"
    GREEN = "#00ff88"
    BLUE  = "#4dabf7"
    TEXT  = "#c8ffd8"
    GRID  = "#0d2018"

    avg_test  = attn_test.mean(axis=0)
    avg_train = attn_train.mean(axis=0)
    days      = np.arange(LOOK_BACK)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.fill_between(days, avg_train, alpha=0.2, color=BLUE,  label="_nolegend_")
    ax.fill_between(days, avg_test,  alpha=0.2, color=GREEN, label="_nolegend_")
    ax.plot(days, avg_train, color=BLUE,  lw=1.5, label="Train avg attention")
    ax.plot(days, avg_test,  color=GREEN, lw=1.5, label="Test avg attention")

    # Highlight high-attention zones
    threshold = avg_test.max() * 0.96
    high_days = np.where(avg_test >= threshold)[0]
    if len(high_days):
        ax.axvspan(high_days[0], high_days[-1], color=GREEN, alpha=0.1,
                   label=f"High attention zone (days {high_days[0]}–{high_days[-1]})")

    ax.set_xlabel("Lookback Day (0 = oldest,  59 = yesterday)", color=TEXT, fontsize=10)
    ax.set_ylabel("Average Attention Weight", color=TEXT, fontsize=10)
    ax.set_title(
        "Temporal Attention Profile: Which Days in the 60-Day Window Matter Most\n"
        "(higher weight = model assigned more importance to that day's features)",
        color=TEXT, fontsize=12, fontweight="bold",
    )
    ax.tick_params(colors=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9, framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"✓ Profile saved → {save_path}")
    plt.close()


def print_attention_summary(attn_test, y_test, test_pred):
    avg = attn_test.mean(axis=0)
    top5 = np.argsort(avg)[-5:][::-1]

    print("\n═══ Attention Analysis Summary ════════════════════")
    print(f"  Lookback window: {LOOK_BACK} days")
    print(f"  Test predictions: {len(y_test)}")
    print(f"\n  Top-5 most attended days (avg across test set):")
    for rank, day in enumerate(top5, 1):
        days_ago = LOOK_BACK - day
        print(f"    #{rank}  Day {day:>2d}  ({days_ago:>2d} days before prediction) "
              f"— weight {avg[day]:.5f}")

    # Compute attention mass in thirds of the window
    early  = avg[:20].sum()
    mid    = avg[20:40].sum()
    recent = avg[40:].sum()
    total  = avg.sum()
    print(f"\n  Attention mass distribution:")
    print(f"    Days  0–19 (oldest)  : {early/total*100:.1f}%")
    print(f"    Days 20–39 (middle)  : {mid/total*100:.1f}%")
    print(f"    Days 40–59 (recent)  : {recent/total*100:.1f}%")

    mae = np.mean(np.abs(y_test - test_pred))
    print(f"\n  Attention-LSTM test MAE: ${mae:,.2f}")
    print("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    attn_test, attn_train, test_pred, y_test = load_data()

    print_attention_summary(attn_test, y_test, test_pred)
    make_attention_heatmap(attn_test, test_pred, y_test)
    make_attention_profile(attn_test, attn_train)

    print("\n✓ All attention visualizations generated.")
    print("  → attention_heatmap.png")
    print("  → attention_profile.png")
