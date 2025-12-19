---
title: "LSTM Networks for Energy Time-Series Forecasting"
author: doha-bounaim
date: 2025-01-05
tags:
  - LSTM
  - deep learning
  - forecasting
  - tutorial
  - time series
---

Long Short-Term Memory (LSTM) networks have become the gold standard for time-series forecasting in energy systems. This tutorial covers the fundamentals and practical implementation for wind-to-hydrogen production forecasting.

<!-- excerpt start -->
Master LSTM networks for energy forecasting with practical examples in wind-to-hydrogen production prediction, from basic concepts to advanced architectures.
<!-- excerpt end -->

## Why LSTM for Energy Forecasting?

Traditional forecasting methods struggle with:
- Long-term dependencies in time series
- Non-linear relationships
- Multiple interacting variables
- Irregular patterns

LSTMs solve these problems through their memory cell architecture.

## LSTM Architecture Basics

### The Memory Cell

An LSTM cell has three gates that control information flow:

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Forget gate
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Input gate
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output gate
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
```

### Gate Functions

1. **Forget Gate**: Decides what to forget from previous state
   - `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`

2. **Input Gate**: Decides what new information to store
   - `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`
   - `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)`

3. **Output Gate**: Decides what to output
   - `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`

## Building an Energy Forecasting Model

### Step 1: Data Preparation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_energy_data(df, sequence_length=24):
    """
    Prepare wind-to-hydrogen production data
    
    Features:
    - Wind speed (m/s)
    - Wind direction (degrees)
    - Temperature (°C)
    - Pressure (hPa)
    - Historical hydrogen production (kg/h)
    """
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, -1])  # Production
    
    return np.array(X), np.array(y), scaler
```

### Step 2: Model Architecture

```python
class EnergyForecastLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction
```

### Step 3: Training Loop

```python
def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                val_loss += loss.item()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, '
                  f'Val Loss = {val_loss:.4f}')
    
    return model
```

## Advanced Architecture: Bidirectional LSTM

Bidirectional LSTMs process sequences in both directions:

```python
class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Key difference
        )
        
        # Note: hidden_size * 2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        prediction = self.fc(bilstm_out[:, -1, :])
        return prediction
```

## Multivariate CNN-BiLSTM Architecture

Our research uses this hybrid architecture for superior performance:

```python
class CNNBiLSTMForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=32, 
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # BiLSTM for temporal dependencies
        self.bilstm = nn.LSTM(
            64, hidden_size, num_layers, 
            batch_first=True, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        
        # CNN expects (batch, features, sequence)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Back to (batch, sequence, features) for LSTM
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Prediction
        prediction = self.fc(context)
        return prediction
```

## Handling Multiple Time Horizons

Predict multiple future time steps:

```python
class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, forecast_horizon=6):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Predict next 6 hours
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions  # Shape: (batch, forecast_horizon)
```

## Evaluation Metrics

```python
def evaluate_forecasts(y_true, y_pred):
    """Calculate key forecasting metrics"""
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R² Score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }
```

## Practical Tips

### 1. Sequence Length Selection
```python
# Test different sequence lengths
for seq_len in [12, 24, 48, 72]:  # hours
    model = train_and_evaluate(seq_len)
    print(f"Sequence length {seq_len}: RMSE = {model.rmse}")
```

### 2. Handling Missing Data
```python
def handle_missing_data(df):
    # Forward fill for short gaps
    df = df.fillna(method='ffill', limit=3)
    
    # Interpolate for longer gaps
    df = df.interpolate(method='linear')
    
    # Mark remaining missing values
    df['is_imputed'] = df.isna().any(axis=1)
    
    return df
```

### 3. Preventing Overfitting
```python
# Early stopping
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

## Real-World Results

From our wind-to-hydrogen forecasting research:

| Metric | Value |
|--------|-------|
| RMSE | 2.3 kg/h |
| MAPE | 4.1% |
| R² | 0.96 |
| Training Time | 45 min |
| Inference Time | <10ms |

## Complete Example

```python
# Load data
df = pd.read_csv('wind_hydrogen_data.csv')
X, y, scaler = prepare_energy_data(df, sequence_length=24)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train), 
    torch.FloatTensor(y_train)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

# Initialize and train model
model = CNNBiLSTMForecaster(input_size=5)
trained_model = train_model(model, train_loader, val_loader)

# Evaluate
predictions = trained_model(torch.FloatTensor(X_test))
metrics = evaluate_forecasts(y_test, predictions.detach().numpy())
print(metrics)
```

## Further Resources

### Papers
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- Our work: "Multivariate CNN-Bi-LSTM temporal production forecasting for distributed wind-to-hydrogen integrated systems"

### Code Libraries
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [TensorFlow Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series)

### Datasets
- [Wind Integration National Dataset (WIND) Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)
- [Open Power System Data](https://open-power-system-data.org/)

## Conclusion

LSTM networks provide powerful tools for energy forecasting. Key takeaways:

✅ Proper data preprocessing is critical
✅ Bidirectional architectures improve accuracy
✅ Hybrid CNN-LSTM models excel at complex patterns
✅ Attention mechanisms enhance interpretability
✅ Regular validation prevents overfitting

Start with a simple LSTM, then progressively add complexity as needed!

---

**Want to discuss LSTM applications in your research?** Contact Doha Bounaim or explore our publications on AI-enabled energy forecasting!
