
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

# Define model
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.net(x)

# Training function
def train_with_batch_size(X_data, y_data, batch_size, run_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    # Class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.array([0, 1, 2, 3]),
                                         y=y_data.numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = DeepNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir=f"runs/batch_{batch_size}")

    # Train loop
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"[Batch {batch_size}] Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

    writer.close()
    print(f"✅ Finished training with batch size {batch_size}")

# Load and preprocess data
data = pd.read_excel("merged_clean_4sheets.xlsx")
X = data.iloc[:, [1, 2, 3]].replace('#', -1)
y = data.iloc[:, 4].replace('#', -1)

X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)
y = pd.to_numeric(y, errors='coerce').fillna(-1)
y = y.replace({-1: 0, 0: 1, 1: 2, 2: 3})

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Run training with two batch sizes
train_with_batch_size(X_tensor, y_tensor, batch_size=16, run_name="batch_16")
train_with_batch_size(X_tensor, y_tensor, batch_size=64, run_name="batch_64")
