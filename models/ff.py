import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as loader
from sklearn.datasets import load_breast_cancer


class NeuralNetwork(nn.Module):
    def __init__(self, layers_sizes: list[int]) -> None:
        self.layers = [
            nn.Linear(size, layers_sizes[i + 1]) for i, size in enumerate(layers_sizes)
        ]

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.randn()


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)

    X = torch.tensor(X)
    y = torch.tensor(y)

    network = NeuralNetwork([32, 32, 32])
    network.fit(X, y)

    y_pred = network.predict(X)
