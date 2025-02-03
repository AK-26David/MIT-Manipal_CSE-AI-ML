import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MNISTNeuralNetwork(nn.Module):
    def __init__(self):
        super(MNISTNeuralNetwork, self).__init__()
        
        # Input layer (784 = 28x28 pixels)
        # First hidden layer
        self.hidden1 = nn.Linear(784, 128)
        
        # Second hidden layer
        self.hidden2 = nn.Linear(128, 64)
        
        # Output layer (10 classes for digits 0-9)
        self.output = nn.Linear(64, 10)
        
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)
        
        # First hidden layer
        x = self.relu(self.hidden1(x))
        
        # Second hidden layer
        x = self.relu(self.hidden2(x))
        
        # Output layer
        x = self.output(x)
        
        return x

def count_parameters(model):
    """
    Count the number of learnable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate_mnist():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False
    )
    
    # Initialize the model
    model = MNISTNeuralNetwork().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Accuracy
    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Confusion Matrix
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)
    for t, p in zip(all_labels, all_preds):
        confusion_matrix[t, p] += 1
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # Print Model Parameters
    param_count = count_parameters(model)
    print(f'\nTotal Learnable Parameters: {param_count}')
    
    return model

# Run the training and evaluation
if __name__ == "__main__":
    model = train_and_evaluate_mnist()