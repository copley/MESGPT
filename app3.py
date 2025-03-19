#!/usr/bin/env python3

import time
import threading
import pandas as pd
import numpy as np

# IB imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

# ML imports (install as needed)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# For neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For Prophet (pip install prophet)
from prophet import Prophet

# For Statsmodels
import statsmodels.api as sm

################################################################################
# 1. IBAPI APP
################################################################################
class IBApp(EWrapper, EClient):
    def __init__(self, ip, port, client_id):
        EClient.__init__(self, self)
        self.ip = ip
        self.port = port
        self.client_id = client_id

        self.historical_data = []
        self.request_completed = False

    def connect_and_run(self):
        self.connect(self.ip, self.port, self.client_id)
        thread = threading.Thread(target=self.run)
        thread.start()

    @iswrapper
    def nextValidId(self, orderId: int):
        """Called automatically when connection is established."""
        print(f"[IBApp] nextValidId: {orderId}")
        self.request_historical_data()

    def request_historical_data(self):
        """Request 1 year of daily data for MES Futures."""
        print("[IBApp] Requesting historical data for MES (1 year, daily bars)...")
        contract = self.create_mes_contract()

        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    def create_mes_contract(self):
        """Per your specs."""
        contract = Contract()
        contract.symbol = "MES"
        contract.secType = "FUT"
        contract.exchange = "CME"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = "20250321"
        contract.localSymbol = "MESH5"
        contract.multiplier = "5"
        return contract

    @iswrapper
    def historicalData(self, reqId, bar):
        self.historical_data.append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    @iswrapper
    def historicalDataEnd(self, reqId, start, end):
        print("[IBApp] Historical data download complete.")
        self.request_completed = True
        self.disconnect()

################################################################################
# 2. UTILITY: RSI CALC
################################################################################
def compute_rsi(series, period=14):
    """Simple RSI calculation."""
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Align final length
    return rsi.reindex(series.index)

################################################################################
# 3. FEATURE ENGINEERING
################################################################################
def feature_engineering(df):
    """
    Minimal feature set:
    - 20-day MA
    - 50-day MA
    - RSI(14)
    - volume
    - next_day_up (classification target)
    """
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = compute_rsi(df["close"], 14)
    
    # Create classification label: 1 if next day's close > today's, else 0
    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)

    # Drop rows with NaN (from rolling, shift, or RSI)
    df.dropna(inplace=True)
    return df

################################################################################
# 4. SINGLE DAY MAIN LOGIC
################################################################################
def main():
    # 1) Connect IB and fetch data
    app = IBApp("127.0.0.1", 7496, client_id=1)
    app.connect_and_run()

    # Wait for data retrieval
    timeout = 60
    start_time = time.time()
    while time.time() - start_time < timeout:
        if app.request_completed:
            break
        time.sleep(1)

    if not app.request_completed:
        print("[Main] Timed out waiting for IB historical data.")
        return

    # 2) Convert to DataFrame
    df = pd.DataFrame(app.historical_data)
    if df.empty:
        print("[Main] No data returned from IB.")
        return

    # Ensure sorting by date
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert date column to a real datetime if it's a string
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n[Main] Fetched {len(df)} daily bars for MES from IB.\n")

    # 3) Feature engineering
    df = feature_engineering(df)

    print("[Main] After feature engineering, sample data:\n", df.tail(5))

    # If not enough rows left, exit
    if len(df) < 50:
        print("[Main] Not enough data after dropping NaNs. Exiting.")
        return

    # 4) Train/Test Split
    feature_cols = ["ma_20", "ma_50", "rsi_14", "volume"]
    X = df[feature_cols].values
    y = df["target"].values

    # Simple chronological split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("\n=============================================")
    print(" Scikit-Learn RandomForest Classification ")
    print("=============================================\n")
    # --------------------------------------------------------------------------
    # 5a) Scikit-Learn Example
    # --------------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RandomForest Test Accuracy: {acc*100:.2f}%")

    # Show last day’s prediction
    last_features = X_test[-1].reshape(1, -1)
    last_prob = rf.predict_proba(last_features)[0][1]
    print(f"Last day probability of 'up': {last_prob:.2f}")
    print("Signal:", "Buy" if last_prob > 0.5 else "No Buy")

    # --------------------------------------------------------------------------
    print("\n=============================================")
    print(" TensorFlow / Keras LSTM (Time-Series) ")
    print("=============================================\n")

    # 5b) TensorFlow LSTM Example
    # For a quick demonstration, let's do a minimal LSTM that uses a rolling window of data.
    # We'll create sequences of length N to predict the label at the end of that sequence.

    sequence_length = 10  # short example
    # We only keep data from after we've dropped NaNs
    # We'll build sequences of shape (sequence_length, num_features)

    def build_sequences(dataX, datay, seq_len=10):
        X_seq, y_seq = [], []
        for i in range(len(dataX) - seq_len):
            X_seq.append(dataX[i : i + seq_len])
            # we'll predict the target at time (i+seq_len-1) or (i+seq_len)
            y_seq.append(datay[i + seq_len - 1])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = build_sequences(X, y, sequence_length)
    # Chronological split again
    seq_split = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:seq_split], X_seq[seq_split:]
    y_train_seq, y_test_seq = y_seq[:seq_split], y_seq[seq_split:]

    model = Sequential()
    model.add(LSTM(32, input_shape=(sequence_length, len(feature_cols))))
    model.add(Dense(1, activation='sigmoid'))  # binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=16, verbose=0)
    loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"LSTM Test Accuracy: {acc*100:.2f}%")

    # Last sequence prediction
    last_seq = X_test_seq[-1].reshape(1, sequence_length, len(feature_cols))
    last_prob_lstm = model.predict(last_seq)[0][0]
    print(f"Last sequence probability of 'up': {last_prob_lstm:.2f}")
    print("Signal:", "Buy" if last_prob_lstm > 0.5 else "No Buy")

    # --------------------------------------------------------------------------
    print("\n=============================================")
    print(" PyTorch LSTM Skeleton ")
    print("=============================================\n")

    # 5c) PyTorch LSTM Example (minimal skeleton)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X_data, y_data, seq_len=10):
            self.X = []
            self.y = []
            for i in range(len(X_data) - seq_len):
                self.X.append(X_data[i : i + seq_len])
                self.y.append(y_data[i + seq_len - 1])
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=32):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out

    # Prepare dataset
    pt_dataset = TimeSeriesDataset(X, y, seq_len=sequence_length)
    pt_split = int(len(pt_dataset) * 0.8)
    pt_train_dataset, pt_test_dataset = torch.utils.data.random_split(
        pt_dataset, [pt_split, len(pt_dataset) - pt_split]
    )

    train_loader = DataLoader(pt_train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(pt_test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model = LSTMModel(input_size=len(feature_cols)).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(pt_model.parameters(), lr=0.001)

    # Train small number of epochs
    for epoch in range(3):
        pt_model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = pt_model(Xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    pt_model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = pt_model(Xb).squeeze()
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    # Convert probabilities to class 0/1
    all_label_preds = (all_preds >= 0.5).astype(int)
    accuracy_pt = np.mean(all_label_preds == all_true)
    print(f"PyTorch LSTM Accuracy: {accuracy_pt*100:.2f}%")

    # Last item
    last_seq_pt = torch.tensor(X[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
    last_prob_pt = pt_model(last_seq_pt).item()
    print(f"Last sequence probability of 'up': {last_prob_pt:.2f}")
    print("Signal:", "Buy" if last_prob_pt > 0.5 else "No Buy")

    # --------------------------------------------------------------------------
    print("\n=============================================")
    print(" Facebook Prophet ")
    print("=============================================\n")

    # 5d) Facebook Prophet (univariate forecast of close price)
    # Prophet expects columns: ds (date), y (value)
    prophet_df = df.rename(columns={"date": "ds", "close": "y"})
    # We'll just do a quick train/test split in date order:
    prop_split = int(len(prophet_df) * 0.8)
    prophet_train = prophet_df.iloc[:prop_split]
    prophet_test = prophet_df.iloc[prop_split:]

    m = Prophet()
    m.fit(prophet_train[["ds", "y"]])

    # Forecast on the entire range (or future)
    future = m.make_future_dataframe(periods=len(prophet_test))
    forecast = m.predict(future)
    # Merge forecast with actual to get test set MSE
    merged = prophet_test.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    mse_prophet = mean_squared_error(merged["y"], merged["yhat"])
    print(f"Prophet Test MSE (close price): {mse_prophet:.2f}")

    # Last day predicted close
    last_pred = merged["yhat"].iloc[-1]
    last_actual = merged["y"].iloc[-1]
    print(f"Prophet: last day predicted close={last_pred:.2f}, actual={last_actual:.2f}")

    # --------------------------------------------------------------------------
    print("\n=============================================")
    print(" Statsmodels ARIMA ")
    print("=============================================\n")

    # 5e) Statsmodels (ARIMA) demonstration for univariate close price
    # We'll do a quick train on the first 80%, test on last 20%
    close_series = df["close"].astype(float)
    train_close = close_series[:prop_split]
    test_close = close_series[prop_split:]

    # Let's try ARIMA(1,1,1) just as an example
    model_arima = sm.tsa.ARIMA(train_close, order=(1,1,1))
    results_arima = model_arima.fit()
    forecast_arima = results_arima.forecast(steps=len(test_close))
    mse_arima = mean_squared_error(test_close, forecast_arima)
    print(f"ARIMA Test MSE (close price): {mse_arima:.2f}")

    print("\nDone. All results printed above.\n")

if __name__ == "__main__":
    main()
