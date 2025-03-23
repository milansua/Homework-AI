
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter

# Define the model
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

def run_training(X_data, y_data, run_name):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    # ✅ FIXED: Ensure `classes` is a numpy array
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2, 3]),
        y=y_data.cpu().numpy()
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = DeepNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == y_train).float().mean().item()

        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"[{run_name}] Epoch {epoch+1}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")

    writer.close()
    print(f"✅ Finished training: {run_name}")

# Load and prepare data
data = pd.read_excel("merged_clean_4sheets.xlsx")
X = data.iloc[:, [1, 2, 3]].replace('#', -1)
y = data.iloc[:, 4].replace('#', -1)

X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)
y = pd.to_numeric(y, errors='coerce').fillna(-1)
y = y.replace({-1: 0, 0: 1, 1: 2, 2: 3})
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Run 1: Without normalization
X_tensor = torch.tensor(X.values, dtype=torch.float32)
run_training(X_tensor, y_tensor, "without_normalization")

# Run 2: With normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
run_training(X_scaled_tensor, y_tensor, "with_normalization")
