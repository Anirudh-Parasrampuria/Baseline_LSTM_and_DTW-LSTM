# ============================================================
# COMMIT 4: Evaluation & Comparison Framework
#
# Loads saved predictions from commits 2 & 3 and produces:
#   - Side-by-side metric table (RMSE, MAE, MAPE, Dir. Acc.)
#   - Prediction vs actual plot (test window)
#   - Residuals analysis plot
#   - Model improvement summary
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ----------------------------------------------------------
# Load saved predictions
# ----------------------------------------------------------
def load_results():
    required = [
        "lstm_test_pred.npy", "lstm_y_test_usd.npy",
        "lstm_test_metrics.npy", "lstm_train_metrics.npy",
        "hybrid_test_pred.npy",
        "hybrid_test_metrics.npy", "hybrid_train_metrics.npy",
        "best_alpha.npy",
    ]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Missing {f}. Run commit2 and commit3 first."
            )

    lstm_pred    = np.load("lstm_test_pred.npy")
    y_test       = np.load("lstm_y_test_usd.npy")
    hybrid_pred  = np.load("hybrid_test_pred.npy")
    best_alpha   = float(np.load("best_alpha.npy")[0])

    metric_keys = ["RMSE ($)", "MAE ($)", "MAPE (%)", "Dir. Acc. (%)"]

    lstm_test_m   = dict(zip(metric_keys, np.load("lstm_test_metrics.npy")))
    lstm_train_m  = dict(zip(metric_keys, np.load("lstm_train_metrics.npy")))
    hybrid_test_m = dict(zip(metric_keys, np.load("hybrid_test_metrics.npy")))
    hybrid_train_m= dict(zip(metric_keys, np.load("hybrid_train_metrics.npy")))

    return (lstm_pred, hybrid_pred, y_test,
            lstm_train_m, lstm_test_m,
            hybrid_train_m, hybrid_test_m,
            best_alpha)


# ----------------------------------------------------------
# Print comparison table
# ----------------------------------------------------------
def print_comparison_table(lstm_train, lstm_test, hybrid_train, hybrid_test):
    metrics = list(lstm_test.keys())
    header = f"{'Metric':<16} {'LSTM Train':>12} {'LSTM Test':>12} {'Hybrid Train':>14} {'Hybrid Test':>13}  {'Δ Test':>10}"
    print("\n" + "═" * len(header))
    print("  Model Comparison: LSTM Baseline vs DTW-LSTM Hybrid")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for m in metrics:
        lt  = lstm_train.get(m, 0)
        lte = lstm_test[m]
        ht  = hybrid_train.get(m, 0)
        hte = hybrid_test[m]

        # Positive delta = improvement (lower error or higher accuracy)
        if "Acc" in m:
            delta = hte - lte
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        else:
            delta = lte - hte
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")

        fmt = ".1f" if "Acc" in m or "MAPE" in m else ",.2f"
        print(
            f"  {m:<14} {lt:>12{fmt}} {lte:>12{fmt}} "
            f"{ht:>14{fmt}} {hte:>13{fmt}}  "
            f"{arrow}{abs(delta):>8{fmt}}"
        )
    print("═" * len(header))


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------
def plot_comparison(lstm_pred, hybrid_pred, y_test, best_alpha, save_path="comparison_plot.png"):
    lstm_res   = y_test - lstm_pred
    hybrid_res = y_test - hybrid_pred
    x = np.arange(len(y_test))

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    C_ACT    = "#58a6ff"
    C_LSTM   = "#f78166"
    C_HYBRID = "#3fb950"
    C_TEXT   = "#c9d1d9"
    C_GRID   = "#21262d"

    ax_style = dict(facecolor="#161b22", tick_params=dict(colors=C_TEXT))

    def style_ax(ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=C_TEXT)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.title.set_color(C_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.grid(True, color=C_GRID, linewidth=0.5)

    # ── Top: Predictions vs Actual ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)
    ax1.plot(x, y_test,      color=C_ACT,    lw=2,   label="Actual ETH Price",     zorder=3)
    ax1.plot(x, lstm_pred,   color=C_LSTM,   lw=1.5, label="LSTM Baseline",        zorder=2, ls="--")
    ax1.plot(x, hybrid_pred, color=C_HYBRID, lw=1.5, label=f"DTW-LSTM Hybrid (α={best_alpha:.2f})", zorder=2)
    ax1.set_title("ETH/USD Price: Actual vs Model Predictions (Test Set)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.8)

    # ── Mid left: Residuals over time ───────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    ax2.axhline(0, color=C_GRID, lw=1)
    ax2.plot(x, lstm_res,   color=C_LSTM,   alpha=0.8, lw=1.2, label="LSTM residuals")
    ax2.plot(x, hybrid_res, color=C_HYBRID, alpha=0.8, lw=1.2, label="Hybrid residuals")
    ax2.set_title("Residuals Over Time")
    ax2.set_ylabel("Prediction Error (USD)")
    ax2.legend(facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.8, fontsize=9)

    # ── Mid right: Residual distribution ────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    bins = np.linspace(
        min(lstm_res.min(), hybrid_res.min()),
        max(lstm_res.max(), hybrid_res.max()),
        20
    )
    ax3.hist(lstm_res,   bins=bins, color=C_LSTM,   alpha=0.6, label="LSTM",   edgecolor="none")
    ax3.hist(hybrid_res, bins=bins, color=C_HYBRID, alpha=0.6, label="Hybrid", edgecolor="none")
    ax3.axvline(0, color="white", lw=1, ls="--")
    ax3.set_title("Residual Distribution")
    ax3.set_xlabel("Error (USD)")
    ax3.legend(facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.8)

    # ── Bottom left: Directional accuracy bar ────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4)
    actual_dir    = np.sign(np.diff(y_test))
    lstm_dir_acc  = np.mean(actual_dir == np.sign(np.diff(lstm_pred)))   * 100
    hybrid_dir_acc= np.mean(actual_dir == np.sign(np.diff(hybrid_pred))) * 100
    bars = ax4.bar(["LSTM Baseline", "DTW-LSTM Hybrid"],
                   [lstm_dir_acc, hybrid_dir_acc],
                   color=[C_LSTM, C_HYBRID], width=0.5)
    ax4.axhline(50, color=C_TEXT, lw=1, ls="--", label="Random guess (50%)")
    ax4.set_ylim(0, 100)
    ax4.set_title("Directional Accuracy (%)")
    ax4.set_ylabel("% Correct Direction")
    ax4.legend(facecolor="#21262d", labelcolor=C_TEXT, framealpha=0.8, fontsize=9)
    for bar, val in zip(bars, [lstm_dir_acc, hybrid_dir_acc]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", color=C_TEXT, fontweight="bold")

    # ── Bottom right: RMSE comparison bar ───────────────
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5)
    lstm_rmse   = np.sqrt(np.mean((y_test - lstm_pred)**2))
    hybrid_rmse = np.sqrt(np.mean((y_test - hybrid_pred)**2))
    bars2 = ax5.bar(["LSTM Baseline", "DTW-LSTM Hybrid"],
                    [lstm_rmse, hybrid_rmse],
                    color=[C_LSTM, C_HYBRID], width=0.5)
    ax5.set_title("Test RMSE (USD) — lower is better")
    ax5.set_ylabel("RMSE (USD)")
    for bar, val in zip(bars2, [lstm_rmse, hybrid_rmse]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"${val:,.0f}", ha="center", va="bottom", color=C_TEXT, fontweight="bold")

    fig.suptitle("DTW-LSTM Hybrid vs LSTM Baseline — ETH/USD Forecasting",
                 color=C_TEXT, fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✓ Comparison plot saved → {save_path}")
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    (lstm_pred, hybrid_pred, y_test,
     lstm_train_m, lstm_test_m,
     hybrid_train_m, hybrid_test_m,
     best_alpha) = load_results()

    print_comparison_table(lstm_train_m, lstm_test_m, hybrid_train_m, hybrid_test_m)

    delta_rmse = lstm_test_m["RMSE ($)"] - hybrid_test_m["RMSE ($)"]
    delta_mape = lstm_test_m["MAPE (%)"] - hybrid_test_m["MAPE (%)"]
    print(f"\n  Summary: Hybrid reduced test RMSE by ${delta_rmse:,.2f} "
          f"and MAPE by {delta_mape:.2f} percentage points vs LSTM baseline.")
    print(f"  Blend: {best_alpha:.0%} LSTM + {1-best_alpha:.0%} DTW analog")

    plot_comparison(lstm_pred, hybrid_pred, y_test, best_alpha,
                    save_path="comparison_plot.png")
