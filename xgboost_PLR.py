import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import TensorDataset, DataLoader

# ─── 1) MAKE RESULT DIRECTORIES ──────────────────────────────────────────────
RESULT_DIR = "result/strategy_cnn"
PLOT_DIR   = os.path.join(RESULT_DIR, "plots")
MODEL_DIR  = os.path.join(RESULT_DIR, "models")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 2) TECHNICAL INDICATOR FUNCTIONS ────────────────────────────────────────
def RSI(data: pd.Series, period: int) -> pd.Series:
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def WR(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (hh - close) / (hh - ll) * -100

def CCI(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - tp_ma) / (0.015 * mad)

def MTM(close: pd.Series, period: int) -> pd.Series:
    return close - close.shift(period)

def DMI(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    up_move   = high.diff()
    down_move = low.diff().abs()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    pdi = 100 * plus_dm.rolling(window=period).mean() / atr
    mdi = 100 * minus_dm.rolling(window=period).mean() / atr
    dx  = 100 * (abs(pdi - mdi) / (pdi + mdi))
    adx = dx.rolling(window=period).mean()
    adxr = (adx + adx.shift(period)) / 2
    return pdi, mdi, adx, adxr

def MACD(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> pd.Series:
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long  = close.ewm(span=long, adjust=False).mean()
    return ema_short - ema_long

# ─── 3) PLR & LABELING FUNCTIONS ─────────────────────────────────────────────
def perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    return np.abs(
        np.cross(line_end - line_start, line_start - point)
    ) / np.linalg.norm(line_end - line_start)

def top_down_plr(series: np.ndarray, epsilon: float) -> list:
    """Recursive Top-Down PLR: returns sorted list of breakpoint indices."""
    def recursive_segment(start_idx: int, end_idx: int, segments: list):
        max_dist = 0.0
        index_of_max = start_idx
        for i in range(start_idx + 1, end_idx):
            dist = perpendicular_distance(
                np.array([i, series[i]]),
                np.array([start_idx, series[start_idx]]),
                np.array([end_idx, series[end_idx]])
            )
            if dist > max_dist:
                max_dist = dist
                index_of_max = i

        if max_dist > epsilon:
            recursive_segment(start_idx, index_of_max, segments)
            recursive_segment(index_of_max, end_idx, segments)
        else:
            segments.append((start_idx, end_idx))

    segments = []
    recursive_segment(0, len(series) - 1, segments)
    idx_set = set()
    for (s, e) in segments:
        idx_set.add(s)
        idx_set.add(e)
    return sorted(idx_set)

def label_from_plr(close_prices: np.ndarray, plr_indices: list) -> np.ndarray:
    """
    Given close_prices[0..N-1] and a sorted list of plr_indices,
    label each plr index as:
       1 (Buy)  if close[curr] < close[prev] & close[curr] < close[next]
       2 (Sell) if close[curr] > close[prev] & close[curr] > close[next]
       0 (Hold) otherwise.
    """
    n = len(close_prices)
    labels = np.zeros(n, dtype=int)
    for i in range(1, len(plr_indices) - 1):
        prev_idx = plr_indices[i - 1]
        curr_idx = plr_indices[i]
        next_idx = plr_indices[i + 1]
        prev_price = close_prices[prev_idx]
        curr_price = close_prices[curr_idx]
        next_price = close_prices[next_idx]
        if (curr_price < prev_price) and (curr_price < next_price):
            labels[curr_idx] = 1  # Buy
        elif (curr_price > prev_price) and (curr_price > next_price):
            labels[curr_idx] = 2  # Sell
    return labels

def find_best_delta(close_prices: np.ndarray,
                    theta_threshold: float = 0.25,
                    delta_grid: np.ndarray = np.linspace(0.01, 0.5, 100)) -> (float, list):
    """
    Finds the smallest delta ∈ delta_grid so that
      (# of PLR breakpoints) / N ≤ theta_threshold.
    Returns (delta, list_of_PLR_indices).
    """
    n = len(close_prices)
    for delta in delta_grid:
        plr_indices = top_down_plr(close_prices, delta)
        theta = len(plr_indices) / n
        if theta <= theta_threshold:
            return delta, plr_indices
    final_delta = delta_grid[-1]
    return final_delta, top_down_plr(close_prices, final_delta)

# ─── 4) MICNN (MULTI-INDICATOR CNN) ────────────────────────────────────────────
class MICNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(MICNN, self).__init__()
        # Input: (batch, 1, 15, 9)
        self.conv1   = nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1))
        self.conv2   = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.pool    = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64 * 15 * 4, 128)  # after pooling → (batch, 64, 15, 4)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 15, 9)
        x = self.dropout(F.relu(self.conv1(x)))   # → (batch, 32, 15, 9)
        x = self.dropout(F.relu(self.conv2(x)))   # → (batch, 64, 15, 9)
        x = self.pool(x)                          # → (batch, 64, 15, 4)
        x = x.view(x.size(0), -1)                 # → (batch, 64*15*4)
        x = self.dropout(F.relu(self.fc1(x)))     # → (batch, 128)
        return self.fc2(x)                        # → (batch, 3)

# ─── 5) BUILD X/Y TENSORS FOR A GIVEN SUBSET ─────────────────────────────────
def build_x_y_tensor_from_dataframe(
    df: pd.DataFrame,
    indicator_cols: list,           # must be exactly 8*15 + 1 = 121 columns (8 indicators × 15 + MACD)
    theta_threshold: float = 0.25
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Returns:
      X_tensor   (N, 1, 15, 9),
      y_tensor   (N,),
      dates      (N,)  as numpy of dtype datetime64[ns],
      close      (N,)  as numpy of float.

    If no rows remain after dropping NaNs, returns four empty arrays.
    """
    required = indicator_cols + ['Close', 'Date', 'TickerName']
    df_clean = df.dropna(subset=required).reset_index(drop=True)

    # If no valid rows remain, return empty arrays
    if df_clean.shape[0] == 0:
        return (
            np.empty((0, 1, 15, 9), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype='datetime64[ns]'),
            np.empty((0,), dtype=float)
        )

    scaler   = MinMaxScaler()
    norm_vals = scaler.fit_transform(df_clean[indicator_cols])  # guaranteed ≥1 row

    N = norm_vals.shape[0]
    X_list = []
    for i in range(N):
        row = norm_vals[i, :]                 # length == 121
        # first 120 entries → reshape into (8,15)
        indicator_matrix = row[:-1].reshape(8, 15)
        # last entry is MACD → replicate 15 times
        macd_row = np.array([row[-1]] * 15).reshape(1, 15)
        # stack → (9,15), then transpose to (15,9)
        full_matrix = np.vstack([indicator_matrix, macd_row])  # shape (9,15)
        reshaped    = full_matrix.T.reshape(1, 15, 9)           # → (1,15,9)
        X_list.append(reshaped)

    X_tensor    = np.stack(X_list, axis=0)      # shape (N, 1, 15, 9)
    close_prices = df_clean['Close'].values     # shape (N,)
    dates       = df_clean['Date'].values.astype('datetime64[ns]')  # shape (N,)

    # PLR labeling on the “close_prices” array
    delta, plr_indices   = find_best_delta(close_prices, theta_threshold=theta_threshold)
    labels               = label_from_plr(close_prices, plr_indices)
    ticker_name          = df_clean['TickerName'].iloc[0]
    plot_plr_labels(close_prices, labels, ticker_name, delta)

    y_tensor = labels.copy()  # shape (N,)
    return X_tensor, y_tensor, dates, close_prices

def plot_plr_labels(close_prices: np.ndarray, labels: np.ndarray, ticker: str, delta: float):
    x = np.arange(len(close_prices))
    buy_idxs  = np.where(labels == 1)[0]
    sell_idxs = np.where(labels == 2)[0]

    plt.figure(figsize=(10, 4))
    plt.plot(x, close_prices, color='black', label="Close")
    plt.scatter(buy_idxs, close_prices[buy_idxs],
                marker='^', color='green', s=50, label="Buy (valley)")
    plt.scatter(sell_idxs, close_prices[sell_idxs],
                marker='v', color='red', s=50, label="Sell (peak)")
    plt.title(f"{ticker} PLR Labels (δ = {delta:.3f})")
    plt.xlabel("Index (day)")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(RESULT_DIR, f"{ticker}_PLR_labels_delta_{delta:.3f}.png")
    plt.savefig(filename)
    plt.close()

# ─── 6) PLOT PLR LABELS & MICNN PREDICTIONS OVER 240 DAYS ─────────────────────
def plot_plr_and_micnn(
    dates: np.ndarray,
    close_prices: np.ndarray,
    plr_labels: np.ndarray,
    cnn_preds: np.ndarray,
    profit: float,
    roi: float,
    ticker: str
):
    """
    Plots:
      - 240-day close curve
      - PLR “Buy” (▲) & “Sell” (▼) on all 240 days
      - MICNN predicted “Buy” (✚) & “Sell” (✖) on all 240 days
      - Annotates total profit and ROI on the title
    """
    df_plot = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Close": close_prices,
        "PLR": plr_labels,
        "CNN": cnn_preds
    })

    plt.figure(figsize=(14, 6))
    plt.plot(df_plot['Date'], df_plot['Close'], color='black', label='Close Price')

    # PLR markers
    buy_plr  = df_plot[df_plot['PLR'] == 1]
    sell_plr = df_plot[df_plot['PLR'] == 2]
    plt.scatter(buy_plr['Date'], buy_plr['Close'], marker='^', color='green', s=60, label='PLR Buy')
    plt.scatter(sell_plr['Date'], sell_plr['Close'], marker='v', color='red',   s=60, label='PLR Sell')

    # CNN markers
    buy_cnn  = df_plot[df_plot['CNN'] == 1]
    sell_cnn = df_plot[df_plot['CNN'] == 2]
    plt.scatter(buy_cnn['Date'], buy_cnn['Close'], marker='P', color='blue',    s=80, label='CNN Pred Buy')
    plt.scatter(sell_cnn['Date'], sell_cnn['Close'], marker='X', color='magenta', s=80, label='CNN Pred Sell')

    plt.title(f"{ticker} – 240 Days: PLR vs. MICNN Preds  |  Profit=${profit:.2f}, ROI={roi*100:.2f}%")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(loc="upper left")
    plt.grid(True)
    filename = os.path.join(PLOT_DIR, f"{ticker}_240days_PLR_and_micnn.png")
    plt.savefig(filename)
    plt.close()

# ─── 7) FULL MICNN WALK-FORWARD OVER 240 DAYS + PROFIT EVAL ───────────────────
def micnn_walkforward_profit(
    df: pd.DataFrame,
    periods: list,
    indicator_names: list,
    train_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float):
    """
    Input:
      - df: 240-row DataFrame with ['Date','Open','High','Low','Close','Volume','TickerName']
      - periods: e.g. list(range(6, 21))
      - indicator_names: e.g. ['RSI','WR','CCI','MTM','PDI','MDI','ADX','ADXR']

    Returns:
      dates_240 (240,), close_240 (240,), plr_labels_240 (240,), micnn_preds_240 (240,),
      total_profit (float), ROI (float)
    """
    df = df.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # 7.1) CALCULATE ALL INDICATORS UP FRONT
    for p in periods:
        df[f"RSI_{p}"]  = RSI(df['Close'], p)
        df[f"WR_{p}"]   = WR(df['High'], df['Low'], df['Close'], p)
        df[f"CCI_{p}"]  = CCI(df['High'], df['Low'], df['Close'], p)
        df[f"MTM_{p}"]  = MTM(df['Close'], p)
        pdi, mdi, adx, adxr = DMI(df['High'], df['Low'], df['Close'], p)
        df[f"PDI_{p}"]  = pdi
        df[f"MDI_{p}"]  = mdi
        df[f"ADX_{p}"]  = adx
        df[f"ADXR_{p}"] = adxr

    df['MACD']    = MACD(df['Close'])
    df['Informer']= (df['Close'].shift(-1) > df['Close']).astype(int)

    # 7.2) COMPUTE PLR ON FULL 240 → get ground-truth labels
    close_arr = df['Close'].values
    delta, plr_idxs = find_best_delta(close_arr, theta_threshold=0.25)
    plr_labels_240 = label_from_plr(close_arr, plr_idxs)

    # 7.3) PREPARE FEATURE LISTS
    feature_cols_cnn  = [f"{ind}_{p}" for ind in indicator_names for p in periods] + ["MACD"]
    feature_cols_full = feature_cols_cnn + ["Informer"]

    micnn_preds = np.zeros(240, dtype=int)
    micnn_preds[0] = 0  # day 0 = Hold, no training data

    # 7.4) WALK-FORWARD LOOP
    for t in range(1, 240):
        # a) Subset training data [0..t-1]
        df_tr = df.iloc[:t].copy().reset_index(drop=True)

        # b) Build training tensors using only feature_cols_cnn (121 columns)
        X_tensor, y_tensor, _, _ = build_x_y_tensor_from_dataframe(
            df_tr,
            indicator_cols=feature_cols_cnn,
            theta_threshold=0.25
        )

        # If no valid training rows, default to Hold
        if X_tensor.shape[0] == 0:
            micnn_preds[t] = 0
            continue

        # If y_tensor has only one unique class, skip oversampling/training
        if len(np.unique(y_tensor)) < 2:
            micnn_preds[t] = 0
            continue

        # c) Oversample the training set
        N_tr     = X_tensor.shape[0]
        X_flat   = X_tensor.reshape(N_tr, -1)
        ros      = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_flat, y_tensor)
        N_res    = X_res.shape[0]
        X_res    = X_res.reshape(N_res, 1, 15, 9)

        # d) DataLoader
        train_ds     = TensorDataset(
            torch.tensor(X_res, dtype=torch.float32),
            torch.tensor(y_res, dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # e) Initialize MICNN
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = MICNN(num_classes=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # f) Train for a fixed number of epochs
        model.train()
        for epoch in range(train_epochs):
            epoch_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # g) Save the model for day t
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"micnn_day{t+1}.pt"))

        # h) Build the test input for day t
        df_test   = df.iloc[[t]].copy().reset_index(drop=True)
        df_concat = pd.concat([df_tr, df_test], axis=0).reset_index(drop=True)

        # Recompute indicators & MACD & Informer on concatenated frame
        for p in periods:
            df_concat.loc[:, f"RSI_{p}"]  = RSI(df_concat['Close'], p)
            df_concat.loc[:, f"WR_{p}"]   = WR(df_concat['High'], df_concat['Low'], df_concat['Close'], p)
            df_concat.loc[:, f"CCI_{p}"]  = CCI(df_concat['High'], df_concat['Low'], df_concat['Close'], p)
            df_concat.loc[:, f"MTM_{p}"]  = MTM(df_concat['Close'], p)
            pdi, mdi, adx, adxr = DMI(df_concat['High'], df_concat['Low'], df_concat['Close'], p)
            df_concat.loc[:, f"PDI_{p}"]  = pdi
            df_concat.loc[:, f"MDI_{p}"]  = mdi
            df_concat.loc[:, f"ADX_{p}"]  = adx
            df_concat.loc[:, f"ADXR_{p}"] = adxr

        df_concat.loc[:, 'MACD']    = MACD(df_concat['Close'])
        df_concat.loc[:, 'Informer'] = (df_concat['Close'].shift(-1) > df_concat['Close']).astype(int)

        # Drop NaNs for CNN features only
        df_clean_all = df_concat.dropna(subset=feature_cols_cnn + ['Close','Date','TickerName']).reset_index(drop=True)

        if df_clean_all.shape[0] == 0:
            micnn_preds[t] = 0
        else:
            norm_vals_all = MinMaxScaler().fit_transform(df_clean_all[feature_cols_cnn])
            N_all_rec = norm_vals_all.shape[0]

            X_list_all = []
            for i in range(N_all_rec):
                row = norm_vals_all[i, :]
                indicator_matrix = row[:-1].reshape(8, 15)
                macd_row         = np.array([row[-1]] * 15).reshape(1, 15)
                full_matrix      = np.vstack([indicator_matrix, macd_row])  # shape (9,15)
                reshaped         = full_matrix.T.reshape(1, 15, 9)
                X_list_all.append(reshaped)

            X_all_recent = np.stack(X_list_all, axis=0)  # shape (N_all_rec, 1, 15, 9)
            X_test_t     = X_all_recent[-1:, ...]        # shape (1, 1, 15, 9)

            if np.isnan(X_test_t).any():
                micnn_preds[t] = 0
            else:
                model.eval()
                with torch.no_grad():
                    Xt = torch.tensor(X_test_t, dtype=torch.float32).to(device)
                    out = model(Xt)
                    micnn_preds[t] = out.argmax(dim=1).cpu().numpy()[0]

    # 7.5) PROFIT SIMULATION
    dates_240 = df['Date'].values
    close_240 = df['Close'].values

    position_open = False
    entry_price   = 0.0
    total_profit  = 0.0
    total_cost    = 0.0

    for t in range(240):
        if micnn_preds[t] == 1 and not position_open:
            # Buy
            entry_price   = close_240[t]
            total_cost   += entry_price
            position_open = True
        elif micnn_preds[t] == 2 and position_open:
            # Sell
            exit_price   = close_240[t]
            profit       = exit_price - entry_price
            total_profit += profit
            position_open = False

    # If still holding at the end, liquidate at last day's close
    if position_open:
        exit_price   = close_240[-1]
        profit       = exit_price - entry_price
        total_profit += profit

    # ROI = total_profit / total_cost
    roi = (total_profit / total_cost) if total_cost != 0 else 0.0

    return dates_240, close_240, plr_labels_240, micnn_preds, total_profit, roi

# ─── 8) RUN FOR EACH TICKER ───────────────────────────────────────────────────
if __name__ == "__main__":
    tickers    = ['AAPL', 'MSFT', 'NVDA']
    periods    = list(range(6, 21))  # 6 through 20
    indicators = ['RSI', 'WR', 'CCI', 'MTM', 'PDI', 'MDI', 'ADX', 'ADXR']

    for ticker in tickers:
        csv_path = os.path.join("data", "price_data", f"{ticker}.csv")
        if not os.path.exists(csv_path):
            print(f"Missing {csv_path}, skipping {ticker}.")
            continue

        df_raw = pd.read_csv(csv_path, parse_dates=["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)

        if df_raw.shape[0] < 240:
            print(f"{ticker} has only {df_raw.shape[0]} rows (<240), skipping.")
            continue

        # Keep only the most recent 240 rows
        df_240 = df_raw.iloc[-240:].reset_index(drop=True)

        (dates_240, close_240,
         plr_labels_240, micnn_preds_240,
         total_profit, roi) = micnn_walkforward_profit(
            df=df_240,
            periods=periods,
            indicator_names=indicators,
            train_epochs=20,
            batch_size=16,
            lr=1e-3
        )

        plot_plr_and_micnn(
            dates=dates_240,
            close_prices=close_240,
            plr_labels=plr_labels_240,
            cnn_preds=micnn_preds_240,
            profit=total_profit,
            roi=roi,
            ticker=ticker
        )
        print(f"  • {ticker}: saved plot → {ticker}_240days_PLR_and_micnn.png")
        print(f"    → Total Profit = ${total_profit:.2f}, ROI = {roi*100:.2f}%\n")

    print("All done!")
