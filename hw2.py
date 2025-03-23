
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

# 🧠 Define the classification model (4 output classes)
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
            nn.Linear(32, 4)  # 4 output neurons
        )
    def forward(self, x):
        return self.net(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepNN().to(device)
model.load_state_dict(torch.load("deep_nn_model.pt", map_location=device))
model.eval()

# Mode selection
choice = input("Select mode: [T]est full dataset or [I]nput your own values? ").strip().lower()

# Mapping for labels
label_map = {-1: 0, 0: 1, 1: 2, 2: 3}
reverse_map = {0: -1, 1: 0, 2: 1, 3: 2}

if choice == "t":
    print("🔁 Running full dataset evaluation...")

    data = pd.read_excel("merged_clean_4sheets.xlsx")
    X = data.iloc[:, [1, 2, 3]].replace('#', -1)
    y = data.iloc[:, 4].replace('#', -1)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)
    y = pd.to_numeric(y, errors='coerce').fillna(-1)
    y = y.replace(label_map)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    _, X_test, _, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        logits = model(X_test)
        test_preds = torch.argmax(logits, dim=1)

    acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

    y_true = [reverse_map[int(v)] for v in y_test.cpu()]
    y_pred = [reverse_map[int(v)] for v in test_preds.cpu()]

    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 0, 1, 2])

    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("🔍 Confusion Matrix")
    plt.grid(False)
    plt.show()

elif choice == "i":
    print("🧠 Manual input mode:")
    try:
        v1 = float(input("Enter value 1: "))
        v2 = float(input("Enter value 2: "))
        v3 = float(input("Enter value 3: "))
        input_tensor = torch.tensor([[v1, v2, v3]], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            predicted_index = torch.argmax(logits, dim=1).item()
            predicted_label = reverse_map[predicted_index]

        label_names = {
            -1: "Error",
             0: "Class 0",
             1: "Class 1",
             2: "Class 2"
        }

        print(f"🎯 Predicted Class: {predicted_label} ({label_names.get(predicted_label, 'Unknown')})")

    except Exception as e:
        print("⚠️ Invalid input. Please enter numeric values only.")

else:
    print("❌ Invalid choice. Please select 'T' or 'I'.")
