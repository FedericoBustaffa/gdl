import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange


class AutorEncoder(nn.Module):
    def __init__(
        self,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()

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
            loss = self.loss_fn(pred, X)

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


def show_reconstructions(model, dataloader, n=8):
    model.eval()

    X, _ = next(iter(dataloader))

    with torch.no_grad():
        recon = model(X)

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


if __name__ == "__main__":
    training_data = MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = MNIST(
        root="datasets",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_subset, _ = random_split(training_data, [100, len(training_data) - 100])
    test_subset, _ = random_split(test_data, [100, len(test_data) - 100])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    ae = AutorEncoder(learning_rate=0.005)
    ae.fit(train_loader, test_loader, max_iter=1000)

    # loss plot
    plt.figure(figsize=(6, 4), dpi=150)
    plt.title("Loss Curve")

    plt.plot(ae.history["train"], label="train")
    plt.plot(ae.history["test"], label="test")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    show_reconstructions(ae, train_loader)
