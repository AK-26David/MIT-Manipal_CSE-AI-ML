import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define the MNIST model (pre-trained)
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained MNIST model (Assume it’s already trained and saved)
model_mnist = MNISTModel()
model_mnist.load_state_dict(torch.load("mnist_model.pth"))  # Load pre-trained weights
model_mnist.eval()

# Modify the model for FashionMNIST (Change the last layer)
class FashionMNISTModel(nn.Module):
    def __init__(self, pretrained_model):
        super(FashionMNISTModel, self).__init__()
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])  # Remove last FC layer
        self.fc_new = nn.Linear(128, 10)  # New classification layer for FashionMNIST

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        x = self.fc_new(x)
        return x

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = "/Users/arnavkarnik/Documents/MIT-Manipal_CSE-AI-ML/Year3/Deep_Learning-Lab/Lab6"

trainset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Transfer Learning: Use pre-trained model
model_fashion = FashionMNISTModel(model_mnist)

# Set optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fashion.parameters(), lr=0.001)

# Train the model on FashionMNIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fashion.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model_fashion.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_fashion(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

# Evaluate on test set
model_fashion.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_fashion(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
