# Lstm_trading
This project is LSTM architecture model for automatic trading Ethereum
Model Architecture
The model is implemented as LSTMModel, utilizing a Long Short-Term Memory (LSTM) architecture for processing sequential trading data.

Input Layer
Input Size: 13 features (open_pct, high_pct, low_pct, close_pct, volume_pct, SMA_10h_pct, EMA_10h_pct, RSI_14h, MACD_pct, MACD_Signal_pct, BB_Upper_pct, BB_Lower_pct, ATR_14h_pct).
Data Format: Sequences of variable length (up to MAX_SEQUENCE_LENGTH=129,600 minutes), normalized to the range [0, 1] using MinMaxScaler.
LSTM Layer
Type: LSTM (Long Short-Term Memory).
Parameters:
hidden_size: 80 (size of the hidden state).
num_layers: 1 (single LSTM layer).
dropout: 0.0 (dropout disabled for simplicity).
batch_first: True (input format: (batch, sequence, features)).
Sequence Handling: Uses pack_padded_sequence and pad_packed_sequence to manage variable-length sequences efficiently.
Fully Connected Layers
Classification (fc_class): Linear(hidden_size, output_size=3) predicts the trade class (0: Buy, 1: Sell, 2: Invalid).
Stop Loss Regression (fc_sl): Linear(hidden_size, 1) predicts the normalized Stop Loss value.
Take Profit Regression (fc_tp): Linear(hidden_size, 1) predicts the normalized Take Profit value.
Output Clamping: sl_out and tp_out are constrained to the range [-10, 10] using torch.clamp.
Output Layer
class_out: Tensor of shape (batch_size, 3), representing probabilities for three classes (Buy, Sell, Invalid).
sl_out: Tensor of shape (batch_size, 1), representing the normalized Stop Loss value.
tp_out: Tensor of shape (batch_size, 1), representing the normalized Take Profit value.
Input Data
Historical Data (historical_data)
Source: SQLite database (bybit1min.db), table candles, symbol ETHUSDT.
Format: Pandas DataFrame with the following columns:
timestamp: Unix timestamp.
open, high, low, close, volume: Price and trading volume data.
Processed Features:
Percentage changes: open_pct, high_pct, low_pct, close_pct, volume_pct.
Technical indicators: SMA_10h, EMA_10h, RSI_14h, MACD, MACD_Signal, BB_Upper, BB_Lower, ATR_14h, and their percentage changes.
Normalization: All features are normalized to the range [0, 1] using MinMaxScaler.
Trade Data (trade_data)
Source: Excel file (val.xlsx), sheet tradesmain.
Format: Pandas DataFrame with the following columns:
start_time: Timestamp of trade initiation.
entry_price: Entry price of the trade.
side: Trade direction (Buy or Sell).
Processed Features:
label: 0 (Buy), 1 (Sell), 2 (Invalid, if entry_price == -1).
stop_loss, take_profit: Normalized values (as percentages of entry price, derived from ATR).
Normalization: Stop Loss and Take Profit values are normalized using MinMaxScaler.
Model Input Format
X_batch: Tensor of shape (batch_size, MAX_SEQUENCE_LENGTH, input_size), containing padded sequences of features.
y_batch: Tensor of shape (batch_size), containing class labels (0, 1, 2).
sl_batch, tp_batch: Tensors of shape (batch_size), containing normalized Stop Loss and Take Profit values.
lengths: Array of sequence lengths for each sample in the batch, used with pack_packed_sequence.
Output Data
class_out: Tensor of shape (batch_size, 3), providing probabilities for the classes (Buy, Sell, Invalid).
sl_out: Tensor of shape (batch_size, 1), representing the normalized Stop Loss value (in [-10, 10]).
tp_out: Tensor of shape (batch_size, 1), representing the normalized Take Profit value (in [-10, 10]).
Loss Functions
Classification: CrossEntropyLoss with class weights to account for class imbalance.
Regression: SmoothL1Loss for both Stop Loss and Take Profit predictions.
Total Loss: Weighted combination: class_weight * class_loss + reg_weight * (sl_loss + tp_loss).
Notes
To use the model in production, denormalize sl_out and tp_out using sl_tp_scaler.inverse_transform to obtain actual Stop Loss and Take Profit values.
The model is optimized for GPU (CUDA) if available, with mixed-precision training (torch.amp.autocast) and gradient scaling (GradScaler) for improved performance and stability.
