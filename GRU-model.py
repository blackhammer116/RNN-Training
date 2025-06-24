import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUWeatherModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        return out

def load_data(file_path='./timeseries-weather-dataset/Weather_Data_1980_2024(hourly).csv'):
    weather_df = pd.read_csv(file_path)
    # Handle missing values (if any)
    if weather_df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        weather_df.fillna(weather_df.mean(), inplace=True)
        print("Missing values handled.")
    else:
        print("No missing values found.")

    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df.set_index('time', inplace=True)
    features_to_scale = weather_df.columns.tolist()

    scaler = MinMaxScaler()
    weather_df[features_to_scale] = scaler.fit_transform(weather_df[features_to_scale])
    display(weather_df.head())
    display(weather_df.info())
    print("\n\n")

    return weather_df

def preprocess_data(weather_df, seq_length=24):

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)].values
            y = data.iloc[i + seq_length].values
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(weather_df, seq_length)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    print(f"Input data shape: {X.shape}")
    print(f"Target data shape: {y.shape}")

    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training data shapes: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shapes: {X_test.shape}, {y_test.shape}")

    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of batches in training dataloader: {len(train_dataloader)}")
    print(f"Number of batches in test dataloader: {len(test_dataloader)}")

    return X_train, y_train, X_test, y_test, train_dataloader, test_dataloader, batch_size


def train_model(model, train_dataloader, epochs=10, lr=0.001):
    print("Starting Training...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_dataloader):.4f}')

    print("Training finished.")


def evaluate_model(model, test_dataloader):
    print("Starting Evaluation...")
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    print(f"Evaluation Results:")
    print(f"  Average Mean Squared Error (MSE): {avg_loss:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")


def make_predictions_and_plot(model, test_dataloader, num_batches_to_plot=1):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
            if i >= num_batches_to_plot:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:, 0], label='Actual')
    plt.plot(predictions[:, 0], label='Predicted')
    plt.title('Weather Feature Prediction vs Actual (First Feature)')
    plt.xlabel('Time Step')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.show()

seq_length = 24 # 24 hours data from the past to predict the next hour
weather_df = load_data()
X_train, y_train, X_test, y_test, train_dataloader, test_dataloader, batch_size = preprocess_data(weather_df, seq_length)

input_size = X_train.shape[2]
hidden_size = 128
output_size = y_train.shape[1]

# Initialize the model
model = GRUWeatherModel(input_size, hidden_size, output_size)
model.to(device)
print(model)

epochs = 10
learning_rate = 0.001
# Start Training
train_model(model, train_dataloader, epochs=epochs, lr=learning_rate)
# Start Eval
evaluate_model(model, test_dataloader)
# Start Prediction
print("Making predictions and plotting results...")
make_predictions_and_plot(model, test_dataloader, num_batches_to_plot=5)