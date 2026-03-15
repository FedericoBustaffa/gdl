import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GaussianMixtureModel:
    def __init__(self, n_categories: int, n_features: int):
        self.n_categories = n_categories

        self.pi = np.random.normal(size=(n_categories,))
        self.mu = np.random.normal(size=(n_categories, n_features))
        self.sigma = np.random.normal(size=(n_categories, n_features))

    def log_likelihood(self, samples):
        N, D = samples.shape
        K = self.n_categories

        diff = samples[:, None, :] - self.mu[None, :, :]
        exponent = np.exp(-(diff**2) / (2 * self.sigma**2))
        gaussian = exponent / (np.sqrt(2 * np.pi) * self.sigma)
        pdf = np.prod(gaussian, axis=2)
        weighted = pdf * self.pi

        return np.sum(np.log(np.sum(weighted, axis=1)))

    def fit(self, samples: np.ndarray):
        """Fits the GMM parameters using the EM algorithm."""
        raise NotImplementedError

    def bic(self, samples: np.ndarray) -> np.ndarray:
        """Computes the BIC score."""
        raise NotImplementedError


if __name__ == "__main__":
    df = pd.read_csv("gaussian_mixture_model/train.csv")
    print(df)

    X = df.to_numpy()
    print(X)

    gmm = GaussianMixtureModel(2, len(df.columns))
    ll = gmm.log_likelihood(X)
    print(f"log likelihood: {ll}")
