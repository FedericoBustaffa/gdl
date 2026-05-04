from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: Sequence[nn.Module],
        decoder: Sequence[nn.Module],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_l1: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_fn = nn.MSELoss()

        # regularization term for sparse AE
        self.lambda_l1 = lambda_l1

        # history tracker
        self.history = {"train": [], "test": []}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_repr = self.encoder(x)
        recon = self.decoder(latent_repr)

        return latent_repr, recon

    def _train_loop(self, dataloader: DataLoader) -> None:
        self.train()
        epoch_loss = 0.0
        for X, _ in dataloader:
            # prediction
            latent, recon = self(X)

            # loss computation
            recon_loss = self.loss_fn(recon, X)
            l1_penalty = latent.abs().mean()
            loss = recon_loss + self.lambda_l1 * l1_penalty

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
                _, recon = self(X)
                loss = self.loss_fn(recon, X)

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
