import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from generative import Autoencoder
from utils import utils

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

    torch.manual_seed(42)

    train_size = 2000
    train_subset, _ = random_split(
        training_data,
        [train_size, len(training_data) - train_size],
    )
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

    test_size = 50
    test_subset, _ = random_split(test_data, [test_size, len(test_data) - test_size])
    test_loader = DataLoader(test_subset, batch_size=50, shuffle=False)

    latent_dim = 64

    encoder = [
        nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 6 * 6, latent_dim),
    ]

    decoder = [
        nn.Linear(latent_dim, 16 * 6 * 6),
        nn.ReLU(),
        nn.Unflatten(1, (16, 6, 6)),
        nn.ConvTranspose2d(16, 32, 3, 2, 0, 0),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, 3, 2, 0, 1),
        nn.Sigmoid(),
    ]

    ae = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        loss_fn=nn.BCELoss(),
        learning_rate=0.01,
        weight_decay=1e-4,
        l1_penalty=0.0,
        noise=0.1,
    )

    ae.fit(train_loader, test_loader, max_iter=200)

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
