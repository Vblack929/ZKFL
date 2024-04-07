import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from federated_learning.data.mnist_processing import get_mnist_dataloaders


def train(model, train_loader, test_loader, epochs=3, lr=0.01, device="mps"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        # test the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {100*correct/total:.2f}%")
    print("Training done")
    # detach the model from the device
    model.to("cpu")

def test(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            acc = 100*correct/total
    print(f"Accuracy: {100*correct/total:.2f}%")
    model.to("cpu")
    return acc
