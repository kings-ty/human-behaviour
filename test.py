import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
# 2. Create synthetic data

num_samples = 500
class0 = torch.randn(num_samples, 2) + torch.tensor([-2.0, -2.0])
labels0 = torch.zeros(num_samples, dtype=torch.long)
class1 = torch.randn(num_samples, 2) + torch.tensor([2.0, 2.0])
labels1 = torch.ones(num_samples, dtype=torch.long)
X = torch.cat([class0, class1])
y = torch.cat([labels0, labels1])
indices = torch.randperm(X.size(0))
X = X[indices]
y = y[indices]
train_size = int(0.8 * X.size(0))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
# 3. Define model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)
model = SimpleClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 4. Training
num_epochs = 50
for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        # Test accuracy
    model.eval()
        correct, total = 0, 0
        with torch.no_grad():
        for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model(Xb)
        _, pred = torch.max(out, 1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader.dataset):.4f}
        "
        f"| Test Acc: {100*correct/total:.2f}%")
        # 5. Using the model
        model.eval()
        new_points = torch.tensor([[0.0, 0.0], [-3.0, -3.0], [3.0, 3.0]]).to(device)
        with torch.no_grad():
        out = model(new_points)
        _, pred = torch.max(out, 1)
        print("Predictions:", pred.cpu())