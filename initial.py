
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Model with configurable initialization
class DeepNN(nn.Module):
    def __init__(self, init_method="random"):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)

        if init_method == "xavier":
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.out]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif init_method == "random":
            pass  # PyTorch uses Kaiming-uniform by default
        else:
            raise ValueError("Unknown init_method")

        self.net = nn.Sequential(
            self.fc1, nn.ReLU(),
            self.fc2, nn.ReLU(),
            self.fc3, nn.ReLU(),
            self.fc4, nn.ReLU(),
            self.out
        )

    def forward(self, x):
        return self.net(x)

# Training function
def train_with_init(X_data, y_data, init_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"runs/init_{init_method}")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.array([0, 1, 2, 3]),
                                         y=y_data.numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = DeepNN(init_method=init_method).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"[{init_method.upper()} INIT] Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    writer.close()
    print(f"✅ Finished training with {init_method.upper()} initialization")

# Load and preprocess data
data = pd.read_excel("merged_clean_4sheets.xlsx")
X = data.iloc[:, [1, 2, 3]].replace('#', -1)
y = data.iloc[:, 4].replace('#', -1)

X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)
y = pd.to_numeric(y, errors='coerce').fillna(-1)
y = y.replace({-1: 0, 0: 1, 1: 2, 2: 3})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Run experiments
train_with_init(X_tensor, y_tensor, init_method="random")
train_with_init(X_tensor, y_tensor, init_method="xavier")
