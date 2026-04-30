from typing import Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange


class Encoder(nn.Module):
    def __init__(
        self,
        layers: Sequence,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_fn = nn.MSELoss()

        # history tracker
        self.history = {"train": [], "test": []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _train_loop(self, dataloader: DataLoader) -> None:
        self.train()
        epoch_loss = 0.0
        for X, _ in dataloader:
            # prediction
            pred = self(X)
            recon_loss = self.loss_fn(pred, X)

            # track training loss
            epoch_loss += loss.item()

            # backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.history["train"].append(epoch_loss / len(dataloader))

    def _test_loop(self, dataloader: DataLoader) -> None:
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for X, _ in dataloader:
                # prediction
                pred = self(X)
                loss = self.loss_fn(pred, X)

                # track test loss
                epoch_loss += loss.item()

        self.history["test"].append(epoch_loss / len(dataloader))

    def fit(
        self,
        training_loader: DataLoader,
        test_loader: DataLoader,
        max_iter: int = 100,
    ) -> None:
        for _ in trange(max_iter, ncols=80, desc="training"):
            self._train_loop(training_loader)
            self._test_loop(test_loader)
