import matplotlib.pyplot as plt
import numpy as np


class UnivariateGaussianMixtureModel:
    def __init__(self, n_centroids: int) -> None:
        self.n_centroids = n_centroids

        self.pi = np.random.random(size=(n_centroids,))
        self.mu = np.random.random(size=(n_centroids,))
        self.sigma = np.random.random(size=(n_centroids,))

    def log_likelihood(self, X: np.ndarray) -> float:
        diff = X[:, None] - self.mu[None, :]
        return np.sum(
            np.log(
                np.sum(
                    self.pi * np.exp(-(diff**2)) / (np.sqrt(2 * np.pi) * self.sigma),
                )
            )
        )


if __name__ == "__main__":
    X1 = np.random.normal(loc=-2, scale=1, size=(100,))
    X2 = np.random.normal(loc=2, scale=1, size=(100,))
    X = np.concatenate([X1, X2])

    print(X)

    plt.hist(X, bins=20, ec="w")
    plt.show()

    ugmm = UnivariateGaussianMixtureModel(2)
    print(f"initial log likelihood: {ugmm.log_likelihood(X):.4f}")

    # ugmm.fit(X)
    # print(f"log likelihood: {ugmm.log_likelihood(X):.4f}")
