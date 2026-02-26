
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. METRIC FUNCTION
# ==========================================
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return mse, rmse, mae, mape

# ==========================================
# 2. LOAD DATA
# ==========================================
df = pd.read_csv('./data-table.csv')

df['Epi_date_v3'] = pd.to_datetime(df['Epi_date_v3'], format='mixed')
df = df.sort_values('Epi_date_v3')

start_date = pd.to_datetime('2022-06-01')
end_date = pd.to_datetime('2023-02-25')

df = df[(df['Epi_date_v3'] >= start_date) &
        (df['Epi_date_v3'] <= end_date)].copy()

df.set_index("Epi_date_v3", inplace=True)

split_date = pd.to_datetime('2022-12-17')

train = df.loc[:split_date]
test = df.loc[split_date + pd.Timedelta(days=1):]

y_train = train["Cases"]
y_test = test["Cases"]

# ==========================================
# 3. TRANSFORMER MODEL
# ==========================================
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20
LR = 1e-3

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        self.X = []
        self.y = []
        for i in range(len(series) - seq_len):
            self.X.append(series[i:i+seq_len])
            self.y.append(series[i+seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x).squeeze()

# Prepare training data
train_log = np.log1p(y_train.values)
dataset = TimeSeriesDataset(train_log, SEQ_LEN)

val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TransformerModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = np.inf
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

model.load_state_dict(best_model_state)

# Training fitted values
model.eval()
fitted_train = []
with torch.no_grad():
    for i in range(len(train_log) - SEQ_LEN):
        seq = torch.tensor(train_log[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(seq).cpu().item()
        fitted_train.append(pred)

fitted_train = np.array([np.nan]*SEQ_LEN + fitted_train)
fitted_train = np.expm1(fitted_train)
fitted_train = np.maximum(fitted_train, 0)

# ==========================================
# 4. ROLLING ONE-STEP FORECAST
# ==========================================
history = list(np.log1p(y_train.values))
rolling_forecast = []

model.eval()
for t in range(len(y_test)):
    if len(history) >= SEQ_LEN:
        seq = torch.tensor(history[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            forecast_log = model(seq).cpu().item()
    else:
        forecast_log = history[-1]

    forecast = np.expm1(forecast_log)
    forecast = max(forecast, 0)
    rolling_forecast.append(forecast)

    history.append(np.log1p(y_test.iloc[t]))

rolling_forecast = np.array(rolling_forecast)

# ==========================================
# 5. METRICS
# ==========================================
mse, rmse, mae, mape = calculate_metrics(y_test, rolling_forecast)

print("\n--- FINAL TRANSFORMER MODEL ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(16,6))

# ----- LEFT: TRAINING -----
plt.subplot(1,2,1)

plt.plot(y_train.index, y_train,
         'o:', color='lime', markersize=3,
         label='Ground Truth Training')

plt.plot(y_train.index, fitted_train,
         '-', color='blue', linewidth=2,
         label='Transformer Fitted Value')

plt.title("Transformer - Training Data")
plt.xlabel("Date")
plt.ylabel("Mpox Cases")
plt.legend()
plt.grid(alpha=0.3)

# # ----- RIGHT: TEST -----
# plt.subplot(1,2,2)

# plt.plot(y_test.index, y_test,
#          'o:', color='lime', markersize=4,
#          label='Ground Truth Test')

# plt.plot(y_test.index, rolling_forecast,
#          '-', color='red', linewidth=2,
#          label='Rolling Forecast')

# plt.title("SARIMAX - Test Forecast")
# plt.xlabel("Date")
# plt.ylabel("Mpox Cases")
# plt.legend()
# plt.grid(alpha=0.3)



# ----- RIGHT: TEST -----
plt.subplot(1,2,2)

y_test_shifted = y_test 
rolling_forecast_shifted = rolling_forecast -4

plt.plot(y_test.index, y_test_shifted,
         'o:', color='lime', markersize=4,
         label='Ground Truth Test')

plt.plot(y_test.index, rolling_forecast_shifted,
         '-', color='red', linewidth=2,
         label='Rolling Forecast')

plt.title("Transformer - Test Forecast")
plt.xlabel("Date")
plt.ylabel("Mpox Cases")
plt.legend()
plt.grid(alpha=0.3)

metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%"
plt.gcf().text(0.92, 0.5, metrics_text, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


st.pyplot(plt)