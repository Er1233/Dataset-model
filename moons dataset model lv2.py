import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class AvdDataset(Dataset):
    def __init__(self, n_samples=500):
        x,y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
        scaler = StandardScaler()

        x = scaler.fit_transform(x)

        self.x = torch.FloatTensor(x)
        self.y  = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class AdvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,8)
        self.fc2 = nn.Linear(8,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_with_validation(model, train_loader, val_loader, epochs=20):
    val_losses = []
    train_losses = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    for epoch in range(epochs):
        train_loss =0
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # FIX: Added .squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss +=loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():  # BONUS: Added no_grad for validation
            for inputs, targets in val_loader:
                outputs = model(inputs).squeeze()  # FIX: Added .squeeze()
                loss = criterion(outputs, targets)
                val_loss +=loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"{epoch+1} / {epochs}")
        print(f"Avg_train_loss: {avg_train_loss:.4f}")
        print(f"Avg_val_loss: {avg_val_loss:.4f}")
    return val_losses, train_losses

def plot_with_validation(val_losses, train_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Training loss", linewidth=2)
    plt.plot(val_losses, label="Validation loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss Vs Validation Loss")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()

def main():
    print("="*50)
    print("Moons Dataset Model")
    print("=="*20)

    #dataset
    dataset = AvdDataset(n_samples=500)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # NEW: Create separate dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    #model
    model = AdvModel()
    print(f"parameters:{sum(p.numel() for p in model.parameters())} ")

    print("\nTraining...")
    val_losses, train_losses = train_with_validation(
        model, train_loader, val_loader, epochs=20
    )

    # 4. Visualize
    print("\nPlotting results...")
    plot_with_validation(val_losses, train_losses)

    # 5. Final validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs).squeeze()  # FIX: Added .squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"\nFinal Validation Accuracy: {accuracy:.2f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()