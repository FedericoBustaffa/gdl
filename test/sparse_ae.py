import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from models import utils
from models.autoencoder import Autoencoder

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

    train_size = 1000
    train_subset, _ = random_split(
        training_data,
        [train_size, len(training_data) - train_size],
    )
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

    test_size = 50
    test_subset, _ = random_split(test_data, [test_size, len(test_data) - test_size])
    test_loader = DataLoader(test_subset, batch_size=50, shuffle=False)

    latent_dim = 2048

    encoder = [
        nn.Flatten(),
        nn.Linear(28 * 28, 1024),
        nn.ReLU(),
        nn.Linear(1024, latent_dim),
    ]

    decoder = [
        nn.Linear(latent_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 28 * 28),
        nn.Unflatten(1, (1, 28, 28)),
        nn.Sigmoid(),
    ]

    ae = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        learning_rate=0.001,
        weight_decay=1e-4,
        lambda_l1=1e-3,
        noise=0.1,
    )

    ae.fit(train_loader, test_loader, max_iter=50)

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

    utils.show_reconstructions(ae, test_loader)
    utils.show_latent_space(ae, train_loader)

    utils.iterative_denoising_grid(
        model=ae,
        dataloader=test_loader,
        noise_levels=[0.1, 0.2, 0.4, 0.6],
        steps=6,
    )
