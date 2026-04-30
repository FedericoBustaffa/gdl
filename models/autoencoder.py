from typing import Sequence

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: Sequence[nn.Module],
        decoder: Sequence[nn.Module],
        learning_rate: float = 1e-3,
        lambda_l1: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.loss_fn = nn.BCELoss()

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


def show_reconstructions(model, dataloader, n=5):
    model.eval()

    X, _ = next(iter(dataloader))

    with torch.no_grad():
        _, recon = model(X)

    X = X[:n]
    recon = recon[:n]

    _, axes = plt.subplots(2, n, figsize=(n * 2, 4))

    for i in range(n):
        # originali
        axes[0, i].imshow(X[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")

        # ricostruzioni
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()


def show_latent_space(model, dataloader):

    model.eval()

    pca = PCA(n_components=2)

    zs = []
    labels = []

    with torch.no_grad():
        for X, y in dataloader:
            z, _ = model(X)
            zs.append(z)
            labels.append(y)

    zs = torch.cat(zs).cpu()
    zs_2d = pca.fit_transform(zs)
    labels = torch.cat(labels).cpu()

    plt.figure(figsize=(6, 5))
    plt.scatter(zs_2d[:, 0], zs_2d[:, 1], c=labels, s=5, cmap="tab10")
    plt.colorbar()
    plt.title("Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()
