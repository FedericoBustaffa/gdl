import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GaussianMixtureModel:
    def __init__(self, n_categories: int, n_features: int):
        self.n_categories = n_categories

        # every cluster equiprobable
        self.pi = np.array([1 / n_categories for _ in range(n_categories)])

        # means all at zero
        self.mu = np.zeros(shape=(n_categories, n_features))

        # variances all at one to prevent initial division by zero
        self.sigma = np.ones(shape=(n_categories, n_features))

    def gaussian(self, samples):
        diff = samples[:, None, :] - self.mu[None, :, :]
        exponent = -(diff**2) / (2 * self.sigma**2)
        gaussian = np.exp(exponent) / (np.sqrt(2 * np.pi) * self.sigma)

        return np.prod(gaussian, axis=2)

    def log_likelihood(self, samples) -> np.ndarray:
        """Computes logP(X|θ)"""
        weighted = self.gaussian(samples) * self.pi

        return np.sum(np.log(np.sum(weighted, axis=1)))

    def fit(self, samples: np.ndarray):
        """Fits the GMM parameters using the EM algorithm."""
        raise NotImplementedError

    def bic(self, samples: np.ndarray) -> np.ndarray:
        """Computes the BIC score."""
        raise NotImplementedError


if __name__ == "__main__":
    df = pd.read_csv("gaussian_mixture_model/train.csv")
    X = df.to_numpy()

    gmm = GaussianMixtureModel(2, len(df.columns))
    ll = gmm.log_likelihood(X)
    print(f"log likelihood: {ll}")
