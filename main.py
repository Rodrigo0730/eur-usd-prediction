import pandas as pd
import sklearn.preprocessing
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import numpy.random as npr

#reproducibility
seed = 42
npr.seed(seed)
torch.manual_seed(seed)
#load data
data = pd.read_csv("eur_usd_data.csv", index_col = 0)
data = data[["Close"]]
data.columns = ["EUR/USD"]
data['EUR/USD'] = pd.to_numeric(data['EUR/USD'], errors='coerce')
data.dropna(inplace=True)

#scaling 
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

x, y = [], []
step = 90
for i in range(len(data) - step -1):
    x.append(data[i:(i + step), 0])
    y.append(data[i + step, 0])

x = np.array(x)
y = np.array(y)

X = x.reshape(x.shape[0], x.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert numpy arrays to PyTorch tensors for model compatibility
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    

model = LSTM(input_size=1, hidden_size=128, num_layers=2, output_size=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

patience = 2
best_loss = np.inf
early_stop = 0

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    scheduler.step()

    # Early Stopping Check
    model.eval()
    with torch.no_grad():
        val_output = model(X_test)
        val_loss = criterion(val_output, y_test)

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    predicted = model(X_test)

predicted_prices = scaler.inverse_transform(predicted.numpy()).flatten()
actual_prices = scaler.inverse_transform(y_test.numpy()).flatten()

# Calculate Error Metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predicted_prices)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R2): {r2:.4f}')



# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('EUR/USD Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()







