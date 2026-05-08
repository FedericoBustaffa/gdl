from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange


def kl_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=1).mean()


class VariationalEncoder(nn.Module):
    def __init__(
        self, hidden_layers: Sequence[nn.Module], output_layer: tuple[int, int]
    ) -> None:
        super().__init__()
        self.core = nn.Sequential(*hidden_layers)
        self.mean = nn.Linear(output_layer[0], output_layer[1])
        self.logvar = nn.Linear(output_layer[0], output_layer[1])

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        z = self.core(x)
        mu = self.mean(z)
        logvar = self.logvar(z)

        return mu, logvar


class VariationalDecoder(nn.Module):
    def __init__(self, hidden_layers: Sequence[nn.Module]) -> None:
        super().__init__()
        self.core = nn.Sequential(*hidden_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.core(z)
        return recon


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: VariationalEncoder,
        decoder: VariationalDecoder,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.beta = beta

        self.recon_loss_fn = nn.BCELoss(reduction="sum")
        self.kl_loss_fn = kl_loss

        self.history = {
            "train": {"recon": [], "kl": [], "total": []},
            "test": {"recon": [], "kl": [], "total": []},
        }

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        latent = mu + std * eps
        recon = self.decoder(latent)

        return mu, logvar, recon

    def _train_loop(self, dataloader: DataLoader) -> None:
        self.train()

        n_samples = len(dataloader)  # for minibatches mean loss
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_loss = 0.0
        for x, _ in dataloader:
            # reconstruction
            mu, logvar, recon = self(x)

            # loss computation
            recon_loss = self.recon_loss_fn(recon, x) / x.size(0)
            kl_loss = self.kl_loss_fn(mu, logvar)
            loss = recon_loss + self.beta * kl_loss

            # track training loss
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_loss += loss.item()

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.history["train"]["recon"].append(epoch_recon_loss / n_samples)
        self.history["train"]["kl"].append(epoch_kl_loss / n_samples)
        self.history["train"]["total"].append(epoch_loss / n_samples)

    def _test_loop(self, dataloader: DataLoader) -> None:
        self.eval()

        n_samples = len(dataloader)  # for minibatches mean loss

        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_loss = 0.0
        with torch.no_grad():
            for x, _ in dataloader:
                # reconstruction
                mu, logvar, recon = self(x)

                # loss computation
                recon_loss = self.recon_loss_fn(recon, x)
                kl_loss = self.kl_loss_fn(mu, logvar)
                loss = recon_loss + kl_loss

                # track training loss
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_loss += loss.item()

        self.history["test"]["recon"].append(epoch_recon_loss / n_samples)
        self.history["test"]["kl"].append(epoch_kl_loss / n_samples)
        self.history["test"]["total"].append(epoch_loss / n_samples)

    def fit(
        self,
        training_loader: DataLoader,
        test_loader: DataLoader,
        max_iter: int = 100,
    ) -> None:
        for _ in trange(max_iter, ncols=80, desc="training"):
            self._train_loop(training_loader)
            self._test_loop(test_loader)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encoder(x)

        latent = mu
        recon = self.decoder(latent)

        return recon

    def latent_code(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encoder(x)
        return mu

    def sample(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            samples = self.decoder(z)

        return samples
