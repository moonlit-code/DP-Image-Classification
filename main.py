import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import matplotlib.pyplot as plt
import numpy as np

# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# 1. Load MNIST Dataset
# --------------------------
def get_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# --------------------------
# 2. Simple CNN (CPSC 340 style)
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------
# 3. Evaluate Accuracy
# --------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

# --------------------------
# 4. DP Training Function
# --------------------------
def train_dp_cnn(
    batch_size: int,
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    epochs: int = 5
):
    train_loader, test_loader = get_mnist(batch_size)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        delta=delta
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=64,
            optimizer=optimizer
        ) as mem_loader:

            for images, labels in mem_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        epsilon = privacy_engine.get_epsilon(delta)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1:2d} | ε = {epsilon:.2f} | Acc = {test_acc:.4f}")

    final_eps = privacy_engine.get_epsilon(delta)
    final_acc = evaluate(model, test_loader)
    return final_eps, final_acc

# --------------------------
# 5. Run multiple DP settings (Ablation)
# --------------------------
if __name__ == "__main__":
    BATCH_SIZE = 128
    # Gradient clip norm: 1.0
    DELTA = 1e-5
    EPOCHS = 5

    noise_multipliers = [0.5, 0.8, 1.2, 1.6, 2.0]
    eps_list = []
    acc_list = []

    print("=== Starting DP Ablation Experiments ===")
    for nm in noise_multipliers:
        print(f"\n--- Running with noise_multiplier = {nm} ---")
        eps, acc = train_dp_cnn(
            batch_size=BATCH_SIZE,
            noise_multiplier=nm,
            delta=DELTA,
            epochs=EPOCHS
        )
        eps_list.append(eps)
        acc_list.append(acc)

    # --------------------------
    # 6. Plot Accuracy vs Epsilon
    # --------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(eps_list, acc_list, marker="o", linewidth=2)
    plt.xlabel("Privacy Budget (ε)")
    plt.ylabel("Test Accuracy")
    plt.title("DP Classification: Accuracy vs Privacy Budget (MNIST)")
    plt.grid(True)
    plt.savefig("accuracy_vs_epsilon.png")
    plt.show()

    # Print results table
    print("\n=== Final Results Table ===")
    print(f"{'Epsilon (ε)':<12} {'Accuracy':<12}")
    for e, a in zip(eps_list, acc_list):
        print(f"{e:<12.2f} {a:<12.4f}")