# LSTM Trading

This project implements an LSTM-based model (`LSTMModel`) for automated Ethereum (ETH) trading, predicting trade actions (Buy, Sell, Invalid) and corresponding Stop Loss and Take Profit levels using sequential market data.

## Model Architecture

The `LSTMModel` leverages a Long Short-Term Memory (LSTM) architecture to process sequential trading data and make predictions.

### Input Layer
- **Input Size**: 13 features:
  - `open_pct`, `high_pct`, `low_pct`, `close_pct`, `volume_pct`
  - `SMA_10h_pct`, `EMA_10h_pct`, `RSI_14h`, `MACD_pct`, `MACD_Signal_pct`, `BB_Upper_pct`, `BB_Lower_pct`, `ATR_14h_pct`
- **Data Format**: Variable-length sequences (up to `MAX_SEQUENCE_LENGTH=129,600` minutes), normalized to `[0, 1]` using `MinMaxScaler`.

### LSTM Layer
- **Type**: Long Short-Term Memory (LSTM).
- **Parameters**:
  - `hidden_size`: 80 (size of the hidden state).
  - `num_layers`: 1 (single LSTM layer).
  - `dropout`: 0.0 (disabled for simplicity).
  - `batch_first`: `True` (input format: `(batch, sequence, features)`).
- **Sequence Handling**: Uses `pack_padded_sequence` and `pad_packed_sequence` to efficiently process variable-length sequences.

### Fully Connected Layers
- **Classification (`fc_class`)**: `Linear(hidden_size, output_size=3)` predicts trade class:
  - `0`: Buy
  - `1`: Sell
  - `2`: Invalid
- **Stop Loss Regression (`fc_sl`)**: `Linear(hidden_size, 1)` predicts normalized Stop Loss.
- **Take Profit Regression (`fc_tp`)**: `Linear(hidden_size, 1)` predicts normalized Take Profit.
- **Output Clamping**: `sl_out` and `tp_out` are constrained to `[-10, 10]` using `torch.clamp`.

### Output Layer
- **class_out**: Tensor of shape `(batch_size, 3)` with probabilities for Buy, Sell, and Invalid classes.
- **sl_out**: Tensor of shape `(batch_size, 1)` with normalized Stop Loss values.
- **tp_out**: Tensor of shape `(batch_size, 1)` with normalized Take Profit values.

## Input Data

### Historical Data (`historical_data`)
- **Source**: SQLite database (`bybit1min.db`), table `candles`, symbol `ETHUSDT`.
- **Format**: Pandas DataFrame with columns:
  - `timestamp`: Unix timestamp
  - `open`, `high`, `low`, `close`, `volume`: Price and trading volume
- **Processed Features**:
  - Percentage changes: `open_pct`, `high_pct`, `low_pct`, `close_pct`, `volume_pct`
  - Technical indicators: `SMA_10h`, `EMA_10h`, `RSI_14h`, `MACD`, `MACD_Signal`, `BB_Upper`, `BB_Lower`, `ATR_14h`, and their percentage changes
- **Normalization**: All features scaled to `[0, 1]` using `MinMaxScaler`.

### Trade Data (`trade_data`)
- **Source**: Excel file (`val.xlsx`), sheet `tradesmain`.
- **Format**: Pandas DataFrame with columns:
  - `start_time`: Timestamp of trade initiation
  - `entry_price`: Entry price of the trade
  - `side`: Trade direction (`Buy` or `Sell`)
- **Processed Features**:
  - `label`: `0` (Buy), `1` (Sell), `2` (Invalid, if `entry_price == -1`)
  - `stop_loss`, `take_profit`: Normalized values (as percentages of entry price, derived from ATR)
- **Normalization**: Stop Loss and Take Profit scaled using `MinMaxScaler`.

### Model Input Format
- **X_batch**: Tensor of shape `(batch_size, MAX_SEQUENCE_LENGTH, input_size)` with padded feature sequences.
- **y_batch**: Tensor of shape `(batch_size)` with class labels (`0`, `1`, `2`).
- **sl_batch**, **tp_batch**: Tensors of shape `(batch_size)` with normalized Stop Loss and Take Profit values.
- **lengths**: Array of sequence lengths per sample for `pack_packed_sequence`.

## Output Data
- **class_out**: `(batch_size, 3)` tensor with probabilities for Buy, Sell, and Invalid classes.
- **sl_out**: `(batch_size, 1)` tensor with normalized Stop Loss values in `[-10, 10]`.
- **tp_out**: `(batch_size, 1)` tensor with normalized Take Profit values in `[-10, 10]`.

## Loss Functions
- **Classification**: `CrossEntropyLoss` with weights to handle class imbalance.
- **Regression**: `SmoothL1Loss` for Stop Loss and Take Profit predictions.
- **Total Loss**: Weighted sum: `class_weight * class_loss + reg_weight * (sl_loss + tp_loss)`.

## Notes
- **Denormalization**: Use `sl_tp_scaler.inverse_transform` on `sl_out` and `tp_out` to obtain actual Stop Loss and Take Profit values for production use.
- **Optimization**: The model supports GPU (CUDA) with mixed-precision training (`torch.amp.autocast`) and gradient scaling (`GradScaler`) for enhanced performance and stability.
