import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Define the models
class CNNClassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True),
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


class CNNClassifier2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True),
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


class CNNClassifier3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(16, 10, bias=True),
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


# Load the MNIST dataset
mnist_trainset = datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)

mnist_testset = datasets.MNIST(root="./data", download=True, train=False, transform=ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=True)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function
loss_fn = nn.CrossEntropyLoss()


# Helper function for training the models
def train_model(model, train_loader, optimizer, num_epochs=6):
    model.to(device)
    model.train()
    total_params = sum(param.numel() for param in model.parameters())

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print(f"Finished Training. Final loss = {loss.item()}, Total params = {total_params}")
    return total_params, loss.item()


# Helper function for testing the models
def test_model(model, test_loader):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


# Train and test CNNClassifier1
model1 = CNNClassifier1()
optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
print("Training CNNClassifier1:")
params1, loss1 = train_model(model1, train_loader, optimizer1)
test_model(model1, test_loader)

# Train and test CNNClassifier2
model2 = CNNClassifier2()
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
print("\nTraining CNNClassifier2:")
params2, loss2 = train_model(model2, train_loader, optimizer2)
test_model(model2, test_loader)

# Train and test CNNClassifier3
model3 = CNNClassifier3()
optimizer3 = optim.SGD(model3.parameters(), lr=0.01)
print("\nTraining CNNClassifier3:")
params3, loss3 = train_model(model3, train_loader, optimizer3)
test_model(model3, test_loader)

# Given losses and params
losses = [0.03902818262577057, 0.08542836457490921, 0.09705353528261185, 0.06044217571616173]
params = [601254, 149798, 38150, 9594]

# Plotting the graph
plt.plot(params, losses, marker='o')
plt.xlabel("Number of Parameters")
plt.ylabel("Loss")
plt.title("Loss vs Number of Parameters")
plt.grid(True)
plt.show()
