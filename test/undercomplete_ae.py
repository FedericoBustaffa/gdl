import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from models.autoencoder import Autoencoder
from models.utils import show_latent_space, show_reconstructions

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

    latent_dim = 16

    encoder = [
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 → 14
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 → 7
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, latent_dim),
    ]

    decoder = [
        nn.Linear(latent_dim, 32 * 7 * 7),
        nn.ReLU(),
        nn.Unflatten(1, (32, 7, 7)),
        nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
        nn.Sigmoid(),
    ]

    ae = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        learning_rate=0.005,
        weight_decay=5e-5,
        lambda_l1=0.0,
    )

    ae.fit(train_loader, test_loader, max_iter=100)

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

    show_reconstructions(ae, test_loader)
    show_latent_space(ae, train_loader)
