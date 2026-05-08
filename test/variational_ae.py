import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from generative import VariationalAutoencoder
from utils import plot


def get_mnist(train_size, test_size, batch_size):
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

    train_subset, _ = random_split(
        training_data,
        [train_size, len(training_data) - train_size],
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    test_subset, _ = random_split(test_data, [test_size, len(test_data) - test_size])
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def kullback_leibler_loss(mu, logsigma):
    kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
    return kl.sum(dim=1).mean()


if __name__ == "__main__":
    torch.manual_seed(9951)

    train_loader, test_loader = get_mnist(
        train_size=5000,
        test_size=100,
        batch_size=128,
    )

    hidden_dim = 128
    latent_dim = 32

    encoder = [
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
    ]

    decoder = [
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 28 * 28),
        nn.Unflatten(1, (1, 28, 28)),
        nn.Sigmoid(),
    ]

    vae = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        reconstruction_loss=nn.BCELoss(reduction="sum"),
        kl_loss=kullback_leibler_loss,
        learning_rate=1e-3,
        weight_decay=1e-5,
        beta=2,
    )

    vae.fit(train_loader, test_loader, max_iter=150)

    # loss plot
    plt.figure(figsize=(6, 4), dpi=150)
    plt.title("Loss Curve")

    # train curves
    plt.plot(vae.history["train"]["recon"], label="train recon")
    plt.plot(vae.history["train"]["kl"], label="train kl")

    # test curves
    plt.plot(vae.history["test"]["recon"], label="test recon")
    plt.plot(vae.history["test"]["kl"], label="test kl")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot.show_reconstructions(vae, train_loader)
    plot.show_reconstructions(vae, test_loader)
    plot.show_latent_space(vae, train_loader)

    vae.eval()
    with torch.no_grad():
        imgs = vae.sample(n_samples=5, latent_dim=latent_dim)

        for img in imgs:
            plt.imshow(img.squeeze(), cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
