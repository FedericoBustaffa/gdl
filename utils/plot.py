import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def show_reconstructions(model, dataloader, n=5):
    model.eval()

    X, _ = next(iter(dataloader))

    with torch.no_grad():
        recon = model.reconstruct(X)

    X = X[:n].cpu()
    recon = recon[:n].cpu()

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
            z = model.latent_code(X)
            zs.append(z.cpu())
            labels.append(y)

    zs = torch.cat(zs).cpu()
    zs_2d = pca.fit_transform(np.array(zs))
    labels = torch.cat(labels).cpu()

    plt.figure(figsize=(6, 4), dpi=200)
    plt.scatter(zs_2d[:, 0], zs_2d[:, 1], c=labels, s=5, cmap="tab10")
    plt.colorbar()
    plt.title("Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.show()


def show_denoising(model, dataloader, noise_level=0.5, n=8):
    model.eval()

    x, _ = next(iter(dataloader))
    x = x[0:1]  # shape: [1, 1, 28, 28]

    noisy_imgs = []
    recon_imgs = []

    with torch.no_grad():
        for _ in range(n):
            noise = torch.randn_like(x) * noise_level
            noisy_x = torch.clamp(x + noise, 0.0, 1.0)

            recon = model.reconstruct(noisy_x)

            noisy_imgs.append(noisy_x.squeeze().cpu())
            recon_imgs.append(recon.squeeze().cpu())

    _, axes = plt.subplots(3, n, figsize=(n * 2, 6))

    for i in range(n):
        axes[0, i].imshow(x.squeeze().cpu(), cmap="gray")
        axes[0, i].axis("off")

        axes[1, i].imshow(noisy_imgs[i], cmap="gray")
        axes[1, i].axis("off")

        axes[2, i].imshow(recon_imgs[i], cmap="gray")
        axes[2, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Noisy")
    axes[2, 0].set_title("Denoised")

    plt.tight_layout()
    plt.show()


def iterative_denoising_grid(model, dataloader, noise_levels, steps):
    model.eval()

    x, _ = next(iter(dataloader))
    x = x[0:1]

    n_rows = len(noise_levels)
    n_cols = steps + 1

    _, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.atleast_2d(axes)

    with torch.no_grad():
        for row, noise_level in enumerate(noise_levels):
            noisy = torch.clamp(
                x + torch.randn_like(x) * noise_level,
                0.0,
                1.0,
            )

            current = noisy
            imgs = [noisy.squeeze().cpu()]

            for _ in range(steps):
                recon = model.reconstruct(current)
                current = torch.clamp(recon, 0.0, 1.0)
                imgs.append(current.squeeze().cpu())

            for col in range(n_cols):
                axes[row, col].imshow(imgs[col], cmap="gray")
                axes[row, col].axis("off")

                if row == 0:
                    axes[row, col].set_title(f"step {col}")

            axes[row, 0].set_ylabel(f"noise={noise_level:.2f}", rotation=90, size=12)

    plt.tight_layout()
    plt.show()
