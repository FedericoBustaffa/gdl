import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


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
    zs_2d = pca.fit_transform(np.array(zs))
    labels = torch.cat(labels).cpu()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(zs_2d[:, 0], zs_2d[:, 1], c=labels, s=5, cmap="tab10")
    plt.colorbar()
    plt.title("Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()
