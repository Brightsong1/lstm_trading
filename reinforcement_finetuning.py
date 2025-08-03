import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import random
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Global fine-tuning parameters
MAX_SEQUENCE_LENGTH = 129600  # Maximum sequence length for input data
NUM_SAMPLES = 1000  # Number of samples for training and validation
TRAIN_RATIO = 0.8  # Ratio of training data
NUM_EPOCHS = 15  # Number of training epochs
BATCH_SIZE = 32  # Batch size for DataLoader
HIDDEN_SIZE = 80  # Hidden size for LSTM
NUM_LAYERS = 1  # Number of LSTM layers
LEARNING_RATE = 0.0001  # Learning rate for AdamW optimizer
WEIGHT_DECAY = 1e-5  # Weight decay for AdamW optimizer
PATIENCE = 3  # Patience for early stopping
INPUT_MODEL_PATH = 'lstm_eth_trading_finetuned_best.pth'  # Path to input model
OUTPUT_MODEL_PATH = 'lstm_eth_trading_finetuned2.pth'  # Path to save fine-tuned model
BEST_MODEL_PATH = 'lstm_eth_trading_finetuned_best2.pth'  # Path to save best model by validation loss

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}, "
      f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, "
      f"cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.enabled else 'disabled'}")

# Disable cuDNN for stability
torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

# Load data
conn = sqlite3.connect('bybit1min.db')
query = "SELECT * FROM candles"
historical_data = pd.read_sql_query(query, conn)
conn.close()
historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='s', errors='coerce')
historical_data = historical_data[historical_data['timestamp'].notna()]
print(f"History range: {historical_data['timestamp'].min()} - {historical_data['timestamp'].max()}")
print(f"Number of records: {len(historical_data)}")
print("First 5 timestamps:\n", historical_data['timestamp'].head(5))
print("Last 5 timestamps:\n", historical_data['timestamp'].tail(5))

# Check required columns
required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
missing_columns = [col for col in required_columns if col not in historical_data.columns]
if missing_columns:
    raise KeyError(f"Missing required columns in data: {missing_columns}")

# Functions for calculating indicators
def calculate_rsi(data, periods=14*60):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12*60, slow=26*60, signal=9*60):
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger(data, window=20*60, num_std=2):
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

def calculate_atr(data, periods=14*60):
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift(1)).abs()
    low_close = (data['low'] - data['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=periods).mean()

# Calculate features
historical_data[['open', 'high', 'low', 'close', 'volume']] = historical_data[['open', 'high', 'low', 'close', 'volume']].clip(lower=0, upper=1e6)
for col in ['open', 'high', 'low', 'close', 'volume']:
    historical_data[f'{col}_pct'] = historical_data[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100) * 100

historical_data['SMA_10h'] = historical_data['close'].rolling(window=10*60).mean().ffill().bfill()
historical_data['EMA_10h'] = historical_data['close'].ewm(span=10*60, adjust=False).mean().ffill().bfill()
historical_data['RSI_14h'] = calculate_rsi(historical_data).ffill().bfill()
historical_data['MACD'], historical_data['MACD_Signal'] = calculate_macd(historical_data)
historical_data['MACD'] = historical_data['MACD'].ffill().bfill()
historical_data['MACD_Signal'] = historical_data['MACD_Signal'].ffill().bfill()
historical_data['BB_Upper'], historical_data['BB_Lower'] = calculate_bollinger(historical_data)
historical_data['BB_Upper'] = historical_data['BB_Upper'].ffill().bfill()
historical_data['BB_Lower'] = historical_data['BB_Lower'].ffill().bfill()
historical_data['ATR_14h'] = calculate_atr(historical_data).ffill().bfill()

# Check ATR
print(f"ATR min: {historical_data['ATR_14h'].min():.2f}, max: {historical_data['ATR_14h'].max():.2f}, mean: {historical_data['ATR_14h'].mean():.2f}")

for col in ['SMA_10h', 'EMA_10h', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR_14h']:
    historical_data[f'{col}_pct'] = historical_data[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100) * 100

# Check indicators
print("Checking indicators for NaN:", historical_data[['SMA_10h_pct', 'EMA_10h_pct', 'RSI_14h', 'MACD_pct', 'MACD_Signal_pct', 'BB_Upper_pct', 'BB_Lower_pct', 'ATR_14h_pct']].isna().sum())
print("Checking indicators for infinite values:", np.isinf(historical_data[['SMA_10h_pct', 'EMA_10h_pct', 'RSI_14h', 'MACD_pct', 'MACD_Signal_pct', 'BB_Upper_pct', 'BB_Lower_pct', 'ATR_14h_pct']]).sum())

# Normalize features
features = ['open_pct', 'high_pct', 'low_pct', 'close_pct', 'volume_pct', 'SMA_10h_pct', 'EMA_10h_pct', 'RSI_14h', 'MACD_pct', 'MACD_Signal_pct', 'BB_Upper_pct', 'BB_Lower_pct', 'ATR_14h_pct']
scaler = MinMaxScaler()
historical_data[features] = scaler.fit_transform(historical_data[features])
print("Checking normalized data for NaN:", historical_data[features].isna().sum())
print("Checking normalized data for infinite values:", np.isinf(historical_data[features]).sum())

# Generate stop-loss and take-profit
def generate_sl_tp(historical_data, num_samples=2000):
    sl_tp_data = []
    for _ in range(num_samples):
        idx = random.randint(14*60, len(historical_data) - 1)
        entry_price = historical_data['close'].iloc[idx]
        atr = historical_data['ATR_14h'].iloc[idx]
        if atr == 0.0:
            atr = 1e-6
        sl_buy = np.clip((4 * atr / entry_price) * 50, a_min=-15, a_max=15)
        tp_buy = np.clip((6 * atr / entry_price) * 50, a_min=-15, a_max=15)
        sl_sell = np.clip((4 * atr / entry_price) * 50, a_min=-15, a_max=15)
        tp_sell = np.clip((6 * atr / entry_price) * 50, a_min=-15, a_max=15)
        sl_tp_data.append([sl_buy, tp_buy])
        sl_tp_data.append([sl_sell, tp_sell])
    sl_tp_data = pd.DataFrame(sl_tp_data, columns=['stop_loss', 'take_profit'])
    print(f"SL range: min={sl_tp_data['stop_loss'].min():.2f}, max={sl_tp_data['stop_loss'].max():.2f}")
    print(f"TP range: min={sl_tp_data['take_profit'].min():.2f}, max={sl_tp_data['take_profit'].max():.2f}")
    return sl_tp_data

sl_tp_data = generate_sl_tp(historical_data, num_samples=2000)
sl_tp_scaler = MinMaxScaler()
sl_tp_scaler.fit(sl_tp_data)
joblib.dump(sl_tp_scaler, 'sl_tp_scaler.pkl')
print("SL/TP scaler created and saved to 'sl_tp_scaler.pkl'")

# Define model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.fc_class = nn.Linear(hidden_size, output_size)
        self.fc_sl = nn.Linear(hidden_size, 1)
        self.fc_tp = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]
        class_out = self.fc_class(out)
        sl_out = torch.clamp(self.fc_sl(out), min=-15.0, max=15.0)
        tp_out = torch.clamp(self.fc_tp(out), min=-15.0, max=15.0)
        return class_out, sl_out, tp_out

# Load model
input_size = len(features)
output_size = 3
model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size)
try:
    state_dict = torch.load(INPUT_MODEL_PATH, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    print(f"Model successfully loaded from '{INPUT_MODEL_PATH}'")
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{INPUT_MODEL_PATH}' not found.")
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")
    raise

model = model.to(device)
model.train()
try:
    model = torch.compile(model)
    print("Model compiled with torch.compile")
except Exception as e:
    print(f"Error in torch.compile: {e}, continuing without compilation")

# Optimizer and loss functions
weights = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32).to(device)
criterion_class = nn.CrossEntropyLoss(weight=weights)
criterion_reg = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = GradScaler('cuda')

def combined_loss(class_out, sl_out, tp_out, y, sl_true, tp_true, class_weight=1.0, reg_weight=0.75):
    class_loss = criterion_class(class_out, y)
    sl_loss = criterion_reg(sl_out, sl_true.unsqueeze(1))
    tp_loss = criterion_reg(tp_out, tp_true.unsqueeze(1))
    total_loss = class_weight * class_loss + reg_weight * (sl_loss + tp_loss)
    return total_loss, class_loss, sl_loss, tp_loss

# Prediction check function
def check_prediction(data, pred_label, pred_sl, pred_tp, entry_price, future_data, sl_tp_scaler):
    pred_sl, pred_tp = sl_tp_scaler.inverse_transform([[pred_sl, pred_tp]])[0]
    pred_sl = np.clip(pred_sl, -15, 15)
    pred_tp = np.clip(pred_tp, -15, 15)

    class_labels = {0: 'Buy', 1: 'Sell', 2: 'Hold'}

    if pred_label == 0:  # Buy
        sl_price = entry_price * (1 - abs(pred_sl) / 20)
        tp_price = entry_price * (1 + abs(pred_tp) / 20)
    elif pred_label == 1:  # Sell
        sl_price = entry_price * (1 + abs(pred_sl) / 20)
        tp_price = entry_price * (1 - abs(pred_tp) / 20)
    else:  # Hold
        sl_price = entry_price * (1 + abs(pred_sl) / 20)
        tp_price = entry_price * (1 + abs(pred_tp) / 20)

    print(f"Entry Price: {entry_price:.2f}, SL Price: {sl_price:.2f}, TP Price: {tp_price:.2f}, "
          f"Pred SL (denorm): {pred_sl:.2f}, Pred TP (denorm): {pred_tp:.2f}, "
          f"Raw SL: {pred_sl:.2f}, Raw TP: {pred_tp:.2f}")

    future_data = future_data[future_data['timestamp'] <= data['timestamp'].iloc[-1] + pd.Timedelta(hours=48)]

    if future_data.empty:
        print("No data in 24-hour window, returning Hold")
        return 2, pred_sl, pred_tp

    print(f"Price range in 24h: min={future_data['low'].min():.2f}, max={future_data['high'].max():.2f}")

    if pred_label == 0:  # Buy
        for _, row in future_data.iterrows():
            if row['high'] >= tp_price:
                print(f"TP reached for Buy: {row['high']:.2f} >= {tp_price:.2f}")
                return 0, pred_sl, pred_tp
            if row['low'] <= sl_price:
                print(f"SL reached for Buy: {row['low']:.2f} <= {sl_price:.2f}")
                return 1, pred_tp, pred_sl
        print("Neither TP nor SL reached for Buy")
        return 2, pred_sl, pred_tp
    elif pred_label == 1:  # Sell
        for _, row in future_data.iterrows():
            if row['low'] <= tp_price:
                print(f"TP reached for Sell: {row['low']:.2f} <= {tp_price:.2f}")
                return 1, pred_sl, pred_tp
            if row['high'] >= sl_price:
                print(f"SL reached for Sell: {row['high']:.2f} >= {sl_price:.2f}")
                return 0, pred_tp, pred_sl
        print("Neither TP nor SL reached for Sell")
        return 2, pred_sl, pred_tp
    else:  # Hold
        return 2, pred_sl, pred_tp

# Prepare data
num_train_samples = int(NUM_SAMPLES * TRAIN_RATIO)
num_val_samples = NUM_SAMPLES - num_train_samples
X_train, y_train, sl_train, tp_train, lengths_train = [], [], [], [], []
X_val, y_val, sl_val, tp_val, lengths_val = [], [], [], [], []

for i in range(NUM_SAMPLES):
    idx = random.randint(MAX_SEQUENCE_LENGTH, len(historical_data) - 24*60)
    start_time = historical_data['timestamp'].iloc[idx]
    end_time = start_time - pd.Timedelta(minutes=MAX_SEQUENCE_LENGTH)
    mask = (historical_data['timestamp'] >= end_time) & (historical_data['timestamp'] <= start_time)
    sequence = historical_data[mask][features].values
    if len(sequence) < 1:
        print(f"Skipping empty sequence at index {idx}, timestamp {start_time}")
        continue

    sequence_tensor = torch.tensor(sequence[-MAX_SEQUENCE_LENGTH:], dtype=torch.float32).to(device)
    padded = torch.zeros(MAX_SEQUENCE_LENGTH, len(features), dtype=torch.float32).to(device)
    seq_len = min(len(sequence), MAX_SEQUENCE_LENGTH)
    padded[-seq_len:] = sequence_tensor[-seq_len:]

    with torch.no_grad(), autocast('cuda', dtype=torch.float16):
        class_out, sl_out, tp_out = model(padded.unsqueeze(0), torch.tensor([seq_len]))
        probabilities = torch.softmax(class_out, dim=1).squeeze().cpu().numpy()
        pred_class = torch.argmax(class_out, dim=1).item()
        pred_sl = sl_out.item()
        pred_tp = tp_out.item()
        class_labels = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
        confidence = probabilities[pred_class] * 100

    future_mask = historical_data['timestamp'] >= start_time
    future_data = historical_data[future_mask]
    entry_price = historical_data['close'].iloc[idx]
    result, sl, tp = check_prediction(historical_data[mask].tail(1), pred_class, pred_sl, pred_tp, entry_price, future_data, sl_tp_scaler)

    print(f"Sample {i+1}: Class={class_labels[pred_class]}, Confidence={confidence:.2f}%, "
          f"SL={sl:.2f}, TP={tp:.2f}, Result={class_labels[result]}, Raw SL={pred_sl:.2f}, Raw TP={pred_tp:.2f}")

    if i < num_train_samples:
        X_train.append(sequence[-MAX_SEQUENCE_LENGTH:])
        y_train.append(result)
        sl_train.append(sl)
        tp_train.append(tp)
        lengths_train.append(seq_len)
    else:
        X_val.append(sequence[-MAX_SEQUENCE_LENGTH:])
        y_val.append(result)
        sl_val.append(sl)
        tp_val.append(tp)
        lengths_val.append(seq_len)

print(f"Collected training sequences: {len(X_train)}, Class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Collected validation sequences: {len(X_val)}, Class distribution:\n{pd.Series(y_val).value_counts()}")

# Create datasets
class FineTuningDataset(Dataset):
    def __init__(self, X, y, stop_losses, take_profits, lengths, max_len):
        self.X = [torch.tensor(x, dtype=torch.float32).contiguous() for x in X]
        self.y = torch.tensor(y, dtype=torch.long).contiguous()
        self.stop_losses = torch.tensor(stop_losses, dtype=torch.float32).contiguous()
        self.take_profits = torch.tensor(take_profits, dtype=torch.float32).contiguous()
        self.lengths = lengths
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.X[idx]
        seq_len = self.lengths[idx]
        padded = torch.zeros(self.max_len, len(features), dtype=torch.float32).contiguous()
        padded[-seq_len:] = seq[-seq_len:]
        return padded, self.y[idx], self.stop_losses[idx], self.take_profits[idx], seq_len

train_dataset = FineTuningDataset(X_train, y_train, sl_train, tp_train, lengths_train, MAX_SEQUENCE_LENGTH)
val_dataset = FineTuningDataset(X_val, y_val, sl_val, tp_val, lengths_val, MAX_SEQUENCE_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    model.train()
    total_loss = 0
    total_class_loss = 0
    total_sl_loss = 0
    total_tp_loss = 0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for i, (X_batch, y_batch, sl_batch, tp_batch, lengths) in enumerate(train_loader):
        X_batch = X_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
        y_batch = y_batch.to(device, non_blocking=True).contiguous()
        sl_batch = sl_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
        tp_batch = tp_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()

        with autocast('cuda', dtype=torch.float16):
            class_out, sl_out, tp_out = model(X_batch, lengths)
            loss, class_loss, sl_loss, tp_loss = combined_loss(class_out, sl_out, tp_out, y_batch, sl_batch, tp_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or infinite loss in batch {i+1}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_sl_loss += sl_loss.item()
        total_tp_loss += tp_loss.item()
        preds = torch.argmax(class_out, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

    train_loss = total_loss / len(train_loader)
    train_class_loss = total_class_loss / len(train_loader)
    train_sl_loss = total_sl_loss / len(train_loader)
    train_tp_loss = total_tp_loss / len(train_loader)
    train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    model.eval()
    val_loss = 0
    val_class_loss = 0
    val_sl_loss = 0
    val_tp_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for X_batch, y_batch, sl_batch, tp_batch, lengths in val_loader:
            X_batch = X_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
            y_batch = y_batch.to(device, non_blocking=True).contiguous()
            sl_batch = sl_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
            tp_batch = tp_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()

            with autocast('cuda', dtype=torch.float16):
                class_out, sl_out, tp_out = model(X_batch, lengths)
                loss, class_loss, sl_loss, tp_loss = combined_loss(class_out, sl_out, tp_out, y_batch, sl_batch, tp_batch)

            val_loss += loss.item()
            val_class_loss += class_loss.item()
            val_sl_loss += sl_loss.item()
            val_tp_loss += tp_loss.item()
            val_preds.extend(torch.argmax(class_out, dim=1).cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_class_loss = val_class_loss / len(val_loader)
    val_sl_loss = val_sl_loss / len(val_loader)
    val_tp_loss = val_tp_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
          f"Training: Loss={train_loss:.4f} (Class={train_class_loss:.4f}, SL={train_sl_loss:.4f}, TP={train_tp_loss:.4f}), "
          f"Accuracy={train_accuracy:.4f}, "
          f"Validation: Loss={val_loss:.4f} (Class={val_class_loss:.4f}, SL={val_sl_loss:.4f}, TP={val_tp_loss:.4f}), "
          f"Accuracy={val_accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
          f"Time: {epoch_time:.2f} sec")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Saved best model based on validation loss")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping: validation loss not improving")
            break

torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
print("Model fine-tuned and saved")
