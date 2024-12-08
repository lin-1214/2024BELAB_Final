import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Hyperparameters
# -------------------------
seq_len = 120       # length of the time series
input_dim = 3       # number of features (alpha, beta, delta)
hidden_dim = 64      # size of LSTM hidden state
num_layers = 1       # number of LSTM layers
num_epochs = 20
batch_size = 16
learning_rate = 0.01

# -------------------------
# Example Data (Replace this with your own data)
# Suppose we have 500 samples for training, each with shape (120, 3)
# and a binary label.
num_train_samples = 340
num_val_samples = 63

# X_train = torch.randn(num_train_samples, seq_len, input_dim)  # shape (N, 120, 3)
# y_train = torch.randint(0, 2, (num_train_samples,)).float()   # shape (N, )

# X_val = torch.randn(num_val_samples, seq_len, input_dim)
# y_val = torch.randint(0, 2, (num_val_samples,)).float()

input_pth = "output"

#load data in output folder, which is csv files be stored in 32 subfolders
X_data = []
y_data = []
X_train = []
y_train = []
X_val = []
y_val = []

#load data
for sub_dir in os.listdir(input_pth):
    sub_pth = os.path.join(input_pth, sub_dir)
    for file in os.listdir(sub_pth):
        if file.endswith(".csv"):
            file_pth = os.path.join(sub_pth, file)
            # print(f"Loading data from {file_pth}")

            #load data
            df = pd.read_csv(file_pth)
            if len(df) < 120:
                print(f"Data length is less than 120, skip this file.")
                continue
            X_data.append(df.iloc[:, 0:3].values)

            #load label
            filename = file.split(".")[0]
            label = filename.split("_")[1] == "1"
            # print(f"label: {label}")
            y_data.append(torch.tensor(label).float())

            # print("Successfully load data.")

#cut the data into training and validation set
X_train = torch.tensor(X_data[:num_train_samples])
y_train = torch.tensor(y_data[:num_train_samples])
X_val = torch.tensor(X_data[num_train_samples:])
y_val = torch.tensor(y_data[num_train_samples:])
print("Successfully load all data.")

print(f"train data check: {y_train.sum()}, {y_train.shape[0]}")
print(f"val data check: {y_val.sum()}, {y_val.shape[0]}")

print("Start training...")

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# Model Definition
# -------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # LSTM output: (batch, seq_len, hidden_dim), (h_n, c_n)
        x = x.float()
        lstm_out, _ = self.lstm(x)  
        # We can take the output of the last time step for classification
        last_step = lstm_out[:, -1, :]  # shape (batch, hidden_dim)
        logits = self.fc(last_step)     # shape (batch, 1)
        probs = self.sigmoid(logits)    # shape (batch, 1)
        return probs

model = LSTMClassifier(input_dim, hidden_dim, num_layers)
model = model.to(device)  # If you have GPU: model.to('cuda')

# -------------------------
# Loss and Optimizer
# -------------------------
criterion = nn.BCELoss()      # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device).float()  # If GPU: X_batch.to('cuda')
        y_batch = y_batch.to(device).unsqueeze(1).float()  # (batch, 1)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        pred = (outputs >= 0.5).float()
        correct += (pred == y_batch).sum().item()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    train_acc = correct / len(train_dataset)
    train_loss = running_loss / len(train_dataset)
    
    # -------------------------
    # Validation Loop
    # -------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(device)  # If GPU: X_val_batch.to('cuda')
            y_val_batch = y_val_batch.to(device).unsqueeze(1)

            outputs = model(X_val_batch)
            v_loss = criterion(outputs, y_val_batch)
            val_loss += v_loss.item() * X_val_batch.size(0)

            # Compute accuracy
            preds = (outputs >= 0.5).float()
            correct += (preds == y_val_batch).sum().item()
            total += y_val_batch.size(0)

    val_loss = val_loss / len(val_dataset)
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
print("Training finished.")

# Save the model
t = time.strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f'lstm_model_{t}.pth')
