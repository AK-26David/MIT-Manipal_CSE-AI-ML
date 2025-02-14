import numpy as np
import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ManualXORVerification:
    def __init__(self):
        # Create neural network
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

        # Extract weights and biases
        self.W1 = self.model[0].weight.detach().numpy()
        self.b1 = self.model[0].bias.detach().numpy()
        self.W2 = self.model[2].weight.detach().numpy()
        self.b2 = self.model[2].bias.detach().numpy()

    def manual_forward(self, X):
        # Reshape input to 2D array if it's 1D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # First layer transformation
        z1 = np.dot(X, self.W1.T) + self.b1
        a1 = 1 / (1 + np.exp(-z1))

        # Second layer transformation
        z2 = np.dot(a1, self.W2.T) + self.b2
        a2 = 1 / (1 + np.exp(-z2))

        return a2

    def pytorch_forward(self, X):
        return self.model(X).detach().numpy()

    def verify_transformations(self):
        # Input data
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

        print("Weights and Biases Details:")
        print("Layer 1 Weights (W1):\n", self.W1)
        print("\nLayer 1 Bias (b1):\n", self.b1)
        print("\nLayer 2 Weights (W2):\n", self.W2)
        print("\nLayer 2 Bias (b2):\n", self.b2)

        print("\nInput Transformations:")
        for i in range(len(X)):
            x_numpy = X[i].numpy()
            manual_out = self.manual_forward(x_numpy)
            pytorch_out = self.pytorch_forward(X[i].unsqueeze(0))

            print(f"\nInput {x_numpy}:")
            print(f"Manual Forward: {manual_out}")
            print(f"PyTorch Forward: {pytorch_out}")
            print(f"Predicted Output: {round(manual_out[0][0])}")


# Run verification
verification = ManualXORVerification()
verification.verify_transformations()