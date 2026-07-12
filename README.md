# Ethereum Price Forecasting with LSTM, Analog Search, and Attention

This project explores next-day **ETH/USD closing-price forecasting** with three deep-learning and pattern-matching approaches:

1. A stacked LSTM baseline
2. A hybrid that blends the LSTM with a nearest-neighbor historical analog forecast
3. An attention-based LSTM that exposes which days in the input window influenced each prediction

The pipeline uses a 60-day lookback window, 13 price and technical-indicator features, and a chronological 80/20 train/test split. Preprocessing is designed to avoid look-ahead leakage: the scaler is fitted only on the training partition.

> **Disclaimer:** This repository is an educational machine-learning project. It is not financial advice, and its outputs should not be used as the sole basis for investment decisions.

## Project overview

Each model predicts the next ETH closing price from the preceding 60 days of market data. The included dataset contains daily ETH/USD OHLCV observations from July 2023 through July 2024.

The feature pipeline generates:

- Close price
- 5-day and 20-day moving averages
- 12-day and 26-day exponential moving averages
- MACD and MACD signal
- 14-period RSI
- Bollinger Band width
- Volume ratio
- Log return
- 5-day and 10-day momentum

Models are evaluated using RMSE, MAE, MAPE, and directional accuracy.

## Models

### Stacked LSTM baseline

The baseline consists of three LSTM layers with dropout, followed by dense regression layers. Training uses early stopping and learning-rate reduction to limit overfitting.

### LSTM + historical analog forecast

The hybrid model searches the training set for the most similar historical 60-day close-price windows. It calculates a distance-weighted next-step forecast from the five nearest windows and blends it with the LSTM prediction. The blend weight is selected by grid search on the training data, with leave-one-out matching used for training forecasts.

Although the project name refers to DTW, the current implementation uses vectorized Euclidean distance over normalized close-price windows as a faster analog-search approximation; it does not compute the Dynamic Time Warping recurrence.

### Attention LSTM

The attention model applies additive temporal attention to the hidden states of a two-layer LSTM. Along with its prediction, it returns a normalized importance weight for every day in the 60-day input window, allowing the model's temporal focus to be visualized.

## Repository structure

```text
.
├── Data_Pipeline.py           # Data loading, indicators, scaling, and sequences
├── LSTM_Baseline.py           # Stacked LSTM training and evaluation
├── DTW-LSTM Hybrid Model.py   # LSTM + nearest-neighbor analog hybrid
├── eval.py                    # Baseline/hybrid comparison and plots
├── attention_lstm.py          # Attention-LSTM training and weight extraction
├── attention_viz.py           # Attention heatmap and profile plots
└── ETH-USD.csv                # Daily ETH/USD OHLCV data
```

Generated models, NumPy arrays, and plots are not required to be present before the first run.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone <your-repository-url>
cd Baseline_LSTM_and_DTW-LSTM

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib tensorflow
```

On Apple silicon, use the TensorFlow package appropriate for your Python and macOS setup if the standard installation is unavailable.

## Usage

Run commands from the repository root.

### 1. Verify the data pipeline

```bash
python Data_Pipeline.py
```

This prints the engineered dataset shape, date range, split date, and sequence dimensions.

### 2. Train the LSTM baseline

```bash
python LSTM_Baseline.py
```

This saves the trained model, predictions, and train/test metrics used by the comparison script.

### 3. Run the hybrid model

```bash
python "DTW-LSTM Hybrid Model.py"
```

If `lstm_baseline.keras` exists, the script reuses it. Otherwise, it trains a baseline model first. It then creates analog forecasts, selects the blend weight, evaluates the hybrid, and saves its results.

### 4. Compare the baseline and hybrid

```bash
python eval.py
```

Run the baseline and hybrid scripts before this step. The evaluation script prints a side-by-side metric table and creates `comparison_plot.png`, including predictions, residuals, RMSE, and directional accuracy.

### 5. Train and inspect the attention model

```bash
python attention_lstm.py
python attention_viz.py
```

The first command trains the attention LSTM and saves its predictions, metrics, and attention weights. The second creates:

- `attention_heatmap.png` — attention weights for individual test predictions
- `attention_profile.png` — average train and test attention across the 60-day window

## Methodology notes

- **Chronological split:** Data is split in time order instead of randomly.
- **Leakage prevention:** `MinMaxScaler` is fitted on training data only.
- **Time-aware training:** Keras training runs with `shuffle=False`.
- **Reproducibility:** NumPy, Python, and TensorFlow use seed `2055`.
- **Early stopping:** The best validation weights are restored automatically.
- **Target:** The next available normalized close price, transformed back to USD for evaluation.

## Outputs

Depending on which scripts are run, the project generates:

- Keras models (`*.keras`)
- Predictions, metrics, and attention weights (`*.npy`)
- Model-comparison and attention visualizations (`*.png`)

These generated artifacts can be large or experiment-specific, so consider adding them to `.gitignore` unless you want to publish a particular trained run.

## Limitations and future work

- The dataset covers roughly one year, which is small for training and evaluating deep sequence models.
- The test partition is correspondingly limited; results should not be treated as evidence of live-trading performance.
- The hybrid currently uses Euclidean nearest-neighbor matching rather than exact DTW.
- Hyperparameters and the hybrid blend are selected using training data rather than a dedicated validation partition.
- Fees, slippage, latency, and trading strategy performance are outside the current evaluation.

Possible extensions include exact or constrained DTW, a longer multi-cycle dataset, rolling walk-forward validation, stronger statistical baselines, uncertainty intervals, and out-of-sample backtesting.

## License

No license file is currently included. Add a license before distributing or accepting contributions—for example, the MIT License for permissive reuse.
