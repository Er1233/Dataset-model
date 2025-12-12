import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SimpleDataset(Dataset):
    def __init__(self, n_sample=500):
        x, y = make_moons(n_samples=n_sample, noise=0.2, random_state=42)

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,8)
        self.fc2 = nn.Linear(8,1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def train_model(model,dataloader, epochs=28):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses =[]
    for epoch in range(epochs):
        epochs_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epochs_loss +=loss.item()
        avg_loss = epochs_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch +1}/{epochs}, Loss: {avg_loss:.4f}")
    return losses
def plot_simple(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary classification")
    plt.show()

def main():
    print("=="*50)
    print("Binary classification model")
    print("="*30)

    #dataset
    dataset = SimpleDataset(n_sample=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset size: {len(dataset)}")

    #model
    model = SimpleModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    #train
    print("\nTraining...")
    losses = train_model(model, dataloader, epochs=20)

    #visualize
    print("\nPlotting results...")
    plot_simple(losses)

    #Test final accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct /total
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        print("\nDone")
if __name__=="__main__":
    main()








