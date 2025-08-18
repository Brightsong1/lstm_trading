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
import joblib
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MAX_SEQUENCE_LENGTH = 129600
MIN_SEQUENCE_LENGTH = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
BINARY_OUTPUT_SIZE = 1
REG_OUTPUT_SIZE = 2
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 3
ACCUMULATION_STEPS = 2
CLASS_WEIGHT = 1.0
REG_WEIGHT = 1.0
GRAD_MAX_NORM = 0.5
DROPOUT = 0.1

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
        label = 0  # Long
        mask = (historical_data['timestamp'] <= row['start_time'])
        atr = historical_data[mask]['ATR_14h'].iloc[-1] if 'ATR_14h' in historical_data else 0.0
        if atr == 0.0:
            atr = 1e-6
        stop_loss = np.clip((row['entry_price'] - 2 * atr) / row['entry_price'] * 100, a_min=-100, a_max=100)
        take_profit = np.clip((row['entry_price'] + 3 * atr) / row['entry_price'] * 100, a_min=-100, a_max=100)
        return label, stop_loss, take_profit
    elif row['side'] == 'Sell':
        label = 1  # Short
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

joblib.dump(scaler, 'features_scaler.pkl')
print("Feature scaler saved to 'features_scaler.pkl'")

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

joblib.dump(sl_tp_scaler, 'sl_tp_scaler.pkl')
print("SL/TP scaler saved to 'sl_tp_scaler.pkl'")

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

y_long = np.where(y == 0, 1, 0)  # Buy=1, else=0
y_short = np.where(y == 1, 1, 0)  # Sell=1, else=0
y_hold = np.where(y == 2, 1, 0)  # Hold=1, else=0

reg_mask = y != 2
X_reg = [X[i] for i in range(len(X)) if reg_mask[i]]
lengths_reg = lengths[reg_mask]
stop_losses_reg = stop_losses[reg_mask]
take_profits_reg = take_profits[reg_mask]

class TradingDataset(Dataset):
    def __init__(self, X, y, lengths, max_len, sl=None, tp=None):
        self.X = [torch.tensor(x, dtype=torch.float32).contiguous() for x in X]
        self.y = torch.tensor(y, dtype=torch.float32).contiguous() if sl is None else None
        self.sl = torch.tensor(sl, dtype=torch.float32).contiguous() if sl is not None else None
        self.tp = torch.tensor(tp, dtype=torch.float32).contiguous() if tp is not None else None
        self.lengths = lengths
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]
        seq_len = self.lengths[idx]
        padded = torch.zeros(self.max_len, len(features), dtype=torch.float32).contiguous()
        padded[-seq_len:] = seq[-seq_len:]
        if self.y is not None:
            return padded, self.y[idx], seq_len
        else:
            return padded, self.sl[idx], self.tp[idx], seq_len

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, out):
        attn_weights = torch.softmax(self.attn(out), dim=1)
        return torch.sum(out * attn_weights, dim=1)

class BinaryLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BinaryLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=DROPOUT, bidirectional=False)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=DROPOUT), num_layers=1)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = out.transpose(0, 1)  # For transformer: (seq_len, batch, hidden)
        out = self.transformer_encoder(out)
        out = out.transpose(0, 1)  # Back to (batch, seq_len, hidden)
        out = self.attention(out)
        return self.fc(out)

class RegressionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RegressionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=DROPOUT, bidirectional=False)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=DROPOUT), num_layers=1)
        self.attention = Attention(hidden_size)
        self.fc_sl = nn.Linear(hidden_size, 1)
        self.fc_tp = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = out.transpose(0, 1)
        out = self.transformer_encoder(out)
        out = out.transpose(0, 1)
        out = self.attention(out)
        sl_out = torch.clamp(self.fc_sl(out), min=-10.0, max=10.0)
        tp_out = torch.clamp(self.fc_tp(out), min=-10.0, max=10.0)
        return sl_out, tp_out

input_size = len(features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_long = BinaryLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, BINARY_OUTPUT_SIZE).to(device, dtype=torch.float32)
model_short = BinaryLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, BINARY_OUTPUT_SIZE).to(device, dtype=torch.float32)
model_hold = BinaryLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, BINARY_OUTPUT_SIZE).to(device, dtype=torch.float32)

model_reg = RegressionLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS).to(device, dtype=torch.float32)

models = {
    'long': model_long,
    'short': model_short,
    'hold': model_hold,
    'reg': model_reg
}

for name, model in models.items():
    try:
        model = torch.compile(model)
        print(f"{name.capitalize()} model compiled with torch.compile")
    except Exception as e:
        print(f"Error in torch.compile for {name}: {e}, continuing without compilation")

criterion_binary = nn.BCEWithLogitsLoss()
criterion_reg = nn.SmoothL1Loss()

def binary_loss(out, y):
    return criterion_binary(out.squeeze(), y)

def reg_loss(sl_out, tp_out, sl_true, tp_true, reg_weight=REG_WEIGHT):
    sl_loss = criterion_reg(sl_out, sl_true.unsqueeze(1))
    tp_loss = criterion_reg(tp_out, tp_true.unsqueeze(1))
    total_loss = reg_weight * (sl_loss + tp_loss)
    return total_loss, sl_loss, tp_loss

optimizers = {
    'long': torch.optim.AdamW(model_long.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    'short': torch.optim.AdamW(model_short.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    'hold': torch.optim.AdamW(model_hold.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    'reg': torch.optim.AdamW(model_reg.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
}

schedulers = {
    'long': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['long'], mode='min', factor=0.5, patience=10),
    'short': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['short'], mode='min', factor=0.5, patience=10),
    'hold': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['hold'], mode='min', factor=0.5, patience=10),
    'reg': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['reg'], mode='min', factor=0.5, patience=10)
}

scalers = {
    'long': GradScaler('cuda'),
    'short': GradScaler('cuda'),
    'hold': GradScaler('cuda'),
    'reg': GradScaler('cuda')
}

for opt in optimizers.values():
    for param_group in opt.param_groups:
        param_group['lr'] = LEARNING_RATE * 0.1

dataset_long = TradingDataset(X, y_long, lengths, MAX_SEQUENCE_LENGTH)
dataset_short = TradingDataset(X, y_short, lengths, MAX_SEQUENCE_LENGTH)
dataset_hold = TradingDataset(X, y_hold, lengths, MAX_SEQUENCE_LENGTH)
dataset_reg = TradingDataset(X_reg, None, lengths_reg, MAX_SEQUENCE_LENGTH, sl=stop_losses_reg, tp=take_profits_reg)

loader_long = DataLoader(dataset_long, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
loader_short = DataLoader(dataset_short, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
loader_hold = DataLoader(dataset_hold, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
loader_reg = DataLoader(dataset_reg, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

loaders = {
    'long': loader_long,
    'short': loader_short,
    'hold': loader_hold,
    'reg': loader_reg
}

for name in ['long', 'short', 'hold', 'reg']:
    model = models[name]
    optimizer = optimizers[name]
    scheduler = schedulers[name]
    scaler = scalers[name]
    loader = loaders[name]
    is_binary = name != 'reg'

    print(f"\nTraining {name.capitalize()} model")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_class_loss = 0 if is_binary else 0
        total_sl_loss = 0 if not is_binary else 0
        total_tp_loss = 0 if not is_binary else 0
        total_correct = 0 if is_binary else 0
        total_samples = 0
        optimizer.zero_grad(set_to_none=True)

        if epoch < WARMUP_EPOCHS:
            lr_scale = (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * lr_scale

        for i, batch in enumerate(loader):
            batch_start_time = time.time()
            if is_binary:
                X_batch, y_batch, lengths = batch
                y_batch = y_batch.to(device, non_blocking=True).contiguous()
            else:
                X_batch, sl_batch, tp_batch, lengths = batch
                sl_batch = sl_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
                tp_batch = tp_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()

            X_batch = X_batch.to(device, non_blocking=True, dtype=torch.float32).contiguous()
            lengths = lengths

            with autocast('cuda', dtype=torch.float16):
                if is_binary:
                    out = model(X_batch, lengths)
                    if torch.isnan(out).any():
                        print(f"NaN in model outputs in batch {i+1} for {name}")
                        continue
                    loss = binary_loss(out, y_batch)
                    class_loss = loss
                else:
                    sl_out, tp_out = model(X_batch, lengths)
                    if torch.isnan(sl_out).any() or torch.isnan(tp_out).any():
                        print(f"NaN in model outputs in batch {i+1} for {name}")
                        continue
                    loss, sl_loss, tp_loss = reg_loss(sl_out, tp_out, sl_batch, tp_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or infinite loss in batch {i+1} for {name}, skipping")
                continue

            loss = loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"NaN or infinite gradient in batch {i+1} for {name}, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * ACCUMULATION_STEPS
            if is_binary:
                total_class_loss += class_loss.item()
                preds = (torch.sigmoid(out) > 0.5).float().squeeze()
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)
            else:
                total_sl_loss += sl_loss.item()
                total_tp_loss += tp_loss.item()
                total_samples += sl_batch.size(0)

            batch_time = time.time() - batch_start_time
            print(f"{name.capitalize()} Batch {i+1}/{len(loader)}, Batch time: {batch_time:.2f} sec")

            del X_batch, lengths
            if is_binary:
                del y_batch, out
            else:
                del sl_batch, tp_batch, sl_out, tp_out
            del loss
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = total_loss / len(loader) if total_loss > 0 else float('nan')
        if is_binary:
            train_class_loss = total_class_loss / len(loader)
            train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            scheduler.step(train_class_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, {name.capitalize()}: Loss={train_loss:.4f} (Class={train_class_loss:.4f}), "
                  f"Accuracy={train_accuracy:.4f}, Epoch time: {time.time() - start_time:.2f} sec, LR={optimizer.param_groups[0]['lr']:.6f}")
        else:
            train_sl_loss = total_sl_loss / len(loader)
            train_tp_loss = total_tp_loss / len(loader)
            scheduler.step(train_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, {name.capitalize()}: Loss={train_loss:.4f} (SL={train_sl_loss:.4f}, TP={train_tp_loss:.4f}), "
                  f"Epoch time: {time.time() - start_time:.2f} sec, LR={optimizer.param_groups[0]['lr']:.6f}")

        if epoch == WARMUP_EPOCHS - 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

        torch.save(model.state_dict(), f'best_lstm_eth_{name}.pth')

for name, model in models.items():
    torch.save(model.state_dict(), f'lstm_eth_{name}_multi.pth')
print("All models saved")
