import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import gc
import os
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MAX_SEQUENCE_LENGTH = 129600
MIN_SEQUENCE_LENGTH = 1
HIDDEN_SIZE = 80
NUM_LAYERS = 1
OUTPUT_SIZE = 3
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 5
ACCUMULATION_STEPS = 4
CLASS_WEIGHT = 1.0
REG_WEIGHT = 1.0
GRAD_MAX_NORM = 0.5

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
print(f"GPU available: {torch.cuda.is_available()}, GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

torch.backends.cudnn.enabled = False
gc.collect()
torch.cuda.empty_cache()

conn = sqlite3.connect('bybit1min.db')
query = "SELECT * FROM candles WHERE symbol = 'ETHUSDT'"
historical_data = pd.read_sql_query(query, conn)
conn.close()

print(f"History range: {historical_data['timestamp'].min()} - {historical_data['timestamp'].max()}")
print(f"History records: {len(historical_data)}")

trade_data = pd.read_excel('val.xlsx', sheet_name='tradesmain')
trade_data = trade_data[(trade_data['start_time'] >= historical_data['timestamp'].min()) &
                        (trade_data['start_time'] <= historical_data['timestamp'].max())].copy()
print(f"Trades after filtering: {len(trade_data)}")

print("Checking trade_data for NaN:", trade_data[['entry_price', 'start_time']].isna().sum())
print("Checking trade_data for infinite values:", np.isinf(trade_data[['entry_price', 'start_time']]).sum())

def get_label_and_targets(row, historical_data):
    if row['entry_price'] == -1:
        return 2, np.nan, np.nan
    elif row['side'] == 'Buy':
        label = 0
        mask = (historical_data['timestamp'] <= row['start_time'])
        atr = historical_data[mask]['ATR_14h'].iloc[-1] if 'ATR_14h' in historical_data else 0.0
        if atr == 0.0:
            atr = 1e-6
        stop_loss = np.clip((row['entry_price'] - 2 * atr) / row['entry_price'] * 100, a_min=-100, a_max=100)
        take_profit = np.clip((row['entry_price'] + 3 * atr) / row['entry_price'] * 100, a_min=-100, a_max=100)
        return label, stop_loss, take_profit
    elif row['side'] == 'Sell':
        label = 1
        mask = (historical_data['timestamp'] <= row['start_time'])
        atr = historical_data[mask]['ATR_14h'].iloc[-1] if 'ATR_14h' in historical_data else 0.0
        if atr == 0.0:
            atr = 1e-6
        stop_loss = np.clip((row['entry_price'] + 2 * atr) / row['entry_price'] * 100, a_min=-100, a_max=100)
        take_profit = np.clip((row['entry_price'] - 3 * atr) / row['entry_price'] * 100, a_min=-100, a_max=100)
        return label, stop_loss, take_profit
    return 2, np.nan, np.nan

trade_data[['label', 'stop_loss', 'take_profit']] = trade_data.apply(
    lambda row: pd.Series(get_label_and_targets(row, historical_data)), axis=1)

print("Checking historical_data for NaN:", historical_data[['open', 'high', 'low', 'close', 'volume']].isna().sum())
print("Checking historical_data for infinite values:", np.isinf(historical_data[['open', 'high', 'low', 'close', 'volume']]).sum())

historical_data[['open', 'high', 'low', 'close', 'volume']] = historical_data[['open', 'high', 'low', 'close', 'volume']].clip(lower=0, upper=1e6)

for col in ['open', 'high', 'low', 'close', 'volume']:
    historical_data[f'{col}_pct'] = historical_data[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100) * 100

historical_data['SMA_10h'] = historical_data['close'].rolling(window=10*60).mean().ffill().bfill()
historical_data['EMA_10h'] = historical_data['close'].ewm(span=10*60, adjust=False).mean().ffill().bfill()

def calculate_rsi(data, periods=14*60):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - (100 / (1 + rs))

historical_data['RSI_14h'] = calculate_rsi(historical_data).ffill().bfill()

def calculate_macd(data, fast=12*60, slow=26*60, signal=9*60):
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

historical_data['MACD'], historical_data['MACD_Signal'] = calculate_macd(historical_data)
historical_data['MACD'] = historical_data['MACD'].ffill().bfill()
historical_data['MACD_Signal'] = historical_data['MACD_Signal'].ffill().bfill()

def calculate_bollinger(data, window=20*60, num_std=2):
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

historical_data['BB_Upper'], historical_data['BB_Lower'] = calculate_bollinger(historical_data)
historical_data['BB_Upper'] = historical_data['BB_Upper'].ffill().bfill()
historical_data['BB_Lower'] = historical_data['BB_Lower'].ffill().bfill()

def calculate_atr(data, periods=14*60):
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift(1)).abs()
    low_close = (data['low'] - data['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=periods).mean()

historical_data['ATR_14h'] = calculate_atr(historical_data).ffill().bfill()

for col in ['SMA_10h', 'EMA_10h', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR_14h']:
    historical_data[f'{col}_pct'] = historical_data[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100) * 100

print("Checking indicators for NaN:", historical_data[['SMA_10h_pct', 'EMA_10h_pct', 'RSI_14h', 'MACD_pct', 'MACD_Signal_pct', 'BB_Upper_pct', 'BB_Lower_pct', 'ATR_14h_pct']].isna().sum())
print("Checking indicators for infinite values:", np.isinf(historical_data[['SMA_10h_pct', 'EMA_10h_pct', 'RSI_14h', 'MACD_pct', 'MACD_Signal_pct', 'BB_Upper_pct', 'BB_Lower_pct', 'ATR_14h_pct']]).sum())

features = ['open_pct', 'high_pct', 'low_pct', 'close_pct', 'volume_pct', 'SMA_10h_pct', 'EMA_10h_pct', 'RSI_14h', 'MACD_pct', 'MACD_Signal_pct', 'BB_Upper_pct', 'BB_Lower_pct', 'ATR_14h_pct']
scaler = MinMaxScaler()
historical_data[features] = scaler.fit_transform(historical_data[features])

print("Checking data after normalization for NaN:", historical_data[features].isna().sum())
print("Checking data after normalization for infinite values:", np.isinf(historical_data[features]).sum())

sl_tp_scaler = MinMaxScaler()
valid_trades = trade_data['entry_price'] != -1
sl_tp_data = trade_data.loc[valid_trades, ['stop_loss', 'take_profit']].replace([np.inf, -np.inf], np.nan).dropna()
sl_tp_data = sl_tp_scaler.fit_transform(sl_tp_data)
trade_data.loc[valid_trades, ['stop_loss', 'take_profit']] = sl_tp_data
median_sl = trade_data.loc[valid_trades, 'stop_loss'].median()
median_tp = trade_data.loc[valid_trades, 'take_profit'].median()
trade_data.loc[~valid_trades, 'stop_loss'] = median_sl
trade_data.loc[~valid_trades, 'take_profit'] = median_tp

print(f"NaN in stop_loss/take_profit after normalization: {trade_data[['stop_loss', 'take_profit']].isna().sum()}")

X, y, stop_losses, take_profits, lengths = [], [], [], [], []
skipped_trades = []

for index, row in trade_data.iterrows():
    start_time = row['start_time']
    end_time = start_time - MAX_SEQUENCE_LENGTH * 60
    mask = (historical_data['timestamp'] >= end_time) & (historical_data['timestamp'] < start_time)
    sequence = historical_data[mask][features].values
    if len(sequence) >= MIN_SEQUENCE_LENGTH:
        X.append(sequence[-MAX_SEQUENCE_LENGTH:])
        y.append(row['label'])
        stop_losses.append(row['stop_loss'])
        take_profits.append(row['take_profit'])
        lengths.append(min(len(sequence), MAX_SEQUENCE_LENGTH))
    else:
        skipped_trades.append((index, len(sequence), start_time))

X = np.array(X, dtype=object)
y = np.array(y, dtype=int)
stop_losses = np.array(stop_losses, dtype=float)
take_profits = np.array(take_profits, dtype=float)
lengths = np.array(lengths, dtype=int)
print(f"Sequences: {len(X)}, Skipped trades: {len(skipped_trades)}")
print(f"Class distribution: \n{pd.Series(y).value_counts()}")
print(f"Checking stop_loss/take_profit for NaN: {np.isnan(stop_losses).sum()}, {np.isnan(take_profits).sum()}")

class TradingDataset(Dataset):
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
        sl_out = torch.clamp(self.fc_sl(out), min=-10.0, max=10.0)
        tp_out = torch.clamp(self.fc_tp(out), min=-10.0, max=10.0)
        return class_out, sl_out, tp_out

input_size = len(features)
model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model = model.to(dtype=torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
try:
    model = torch.compile(model)
    print("Model compiled with torch.compile")
except Exception as e:
    print(f"Error in torch.compile: {e}, continuing without compilation")

class_counts = pd.Series(y).value_counts()
weights = 1.0 / torch.tensor([class_counts.get(i, 1) for i in range(3)], dtype=torch.float32)
weights = weights / weights.sum()
weights = weights.to(device)
criterion_class = nn.CrossEntropyLoss(weight=weights)
criterion_reg = nn.SmoothL1Loss()

def combined_loss(class_out, sl_out, tp_out, y, sl_true, tp_true, class_weight=CLASS_WEIGHT, reg_weight=REG_WEIGHT):
    class_loss = criterion_class(class_out, y)
    sl_loss = criterion_reg(sl_out, sl_true.unsqueeze(1))
    tp_loss = criterion_reg(tp_out, tp_true.unsqueeze(1))
    total_loss = class_weight * class_loss + reg_weight * (sl_loss + tp_loss)
    return total_loss, class_loss, sl_loss, tp_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scaler = GradScaler('cuda')

for param_group in optimizer.param_groups:
    param_group['lr'] = LEARNING_RATE * 0.1

train_dataset = TradingDataset(X, y, stop_losses, take_profits, lengths, MAX_SEQUENCE_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

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

    if epoch < WARMUP_EPOCHS:
        lr_scale = (epoch + 1) / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE * lr_scale

    for i, (X_batch, y_batch, sl_batch, tp_batch, lengths) in enumerate(train_loader):
        batch_start_time = time.time()
        X_batch = X_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
        y_batch = y_batch.to(device, non_blocking=True).contiguous()
        sl_batch = sl_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
        tp_batch = tp_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
        lengths = lengths

        with autocast('cuda', dtype=torch.float16):
            class_out, sl_out, tp_out = model(X_batch, lengths)

            if torch.isnan(class_out).any() or torch.isnan(sl_out).any() or torch.isnan(tp_out).any():
                print(f"NaN in model outputs in batch {i+1}")
                continue

            loss, class_loss, sl_loss, tp_loss = combined_loss(class_out, sl_out, tp_out, y_batch, sl_batch, tp_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or infinite loss in batch {i+1}, skipping")
            continue

        loss = loss / ACCUMULATION_STEPS
        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"NaN or infinite gradient in batch {i+1}, skipping")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * ACCUMULATION_STEPS
        total_class_loss += class_loss.item()
        total_sl_loss += sl_loss.item()
        total_tp_loss += tp_loss.item()
        preds = torch.argmax(class_out, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

        batch_time = time.time() - batch_start_time
        print(f"Batch {i+1}/{len(train_loader)}, Batch time: {batch_time:.2f} sec")

        del X_batch, y_batch, sl_batch, tp_batch, class_out, sl_out, tp_out, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss = total_loss / len(train_loader) if total_loss > 0 else float('nan')
    train_class_loss = total_class_loss / len(train_loader)
    train_sl_loss = total_sl_loss / len(train_loader)
    train_tp_loss = total_tp_loss / len(train_loader)
    train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    scheduler.step(train_loss)

    if epoch == WARMUP_EPOCHS - 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
          f"Training: Loss={train_loss:.4f} (Class={train_class_loss:.4f}, SL={train_sl_loss:.4f}, TP={train_tp_loss:.4f}), "
          f"Accuracy={train_accuracy:.4f}, Epoch time: {epoch_time:.2f} sec, LR={optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), 'best_lstm_eth_trading.pth')

torch.save(model.state_dict(), 'lstm_eth_trading_multi.pth')
print("Model saved")

