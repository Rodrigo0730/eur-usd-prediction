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

#load data
data = pd.read_csv("eur_usd_data.csv", index_col = 0)
data = data[["Close"]]
data.columns = ["EUR/USD"]
data['EUR/USD'] = pd.to_numeric(data['EUR/USD'], errors='coerce')
data.dropna(inplace=True)

#scaling 
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

x, y = [], []
step = 30
for i in range(len(data) - step -1):
    x.append(data[i:(i + step), 0])
    y.append(data[i + step, 0])

x = np.array(x)
y = np.array(y)

X = x.reshape(x.shape[0], x.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=5, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM()

criterion = nn.SmoothL1Loss()
optimizer = optim.Adamax(model.parameters(), lr=0.03)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)

predicted_prices = scaler.inverse_transform(predicted.numpy()).flatten()
actual_prices = scaler.inverse_transform(y_test_tensor.numpy()).flatten()

# 1. Calculate error metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r_squared = r2_score(actual_prices, predicted_prices)

# 2. Calculate residuals
residuals = actual_prices - predicted_prices

# 3. Perform Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(residuals)

# 4. Perform Shapiro-Wilk test for normality
shapiro_stat, shapiro_p_value = shapiro(residuals)

# 5. Print the results
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R2): {r_squared:.4f}')
print(f'Durbin-Watson Statistic: {dw_stat:.4f}')
print(f'Shapiro-Wilk Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p_value:.4f}')



"""
plt.figure(figsize=(25, 8))
plt.plot(actual_prices, label='Actual Prices', color='blue')
plt.plot(predicted_prices, label='Predicted Prices', color='orange')
plt.title('EUR/USD Exchange Rate Prediction')
plt.xlabel('Time')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()
"""




