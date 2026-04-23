import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dl
from sklearn.datasets import load_breast_cancer
from torch import tensor


class MLP(nn.Module):
    def __init__(self, layers_sizes: list[int]) -> None:
        self.layers = [
            nn.Linear(size, layers_sizes[i + 1]) for i, size in enumerate(layers_sizes)
        ]

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.randn()


if __name__ == "__main__":
    data = load_breast_cancer()

    X = torch.tensor(data.data)
    y = torch.tensor(data.target)
