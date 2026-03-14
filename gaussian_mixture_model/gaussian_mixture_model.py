import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GaussianMixtureModel:
    def __init__(self, n_categories: int, n_features: int):
        self.n_categories = n_categories

        self.pi = None  # Shape: (n_categories,)
        self.mu = None  # Shape: (n_categories, n_features)
        self.sigma = None  # Shape: (n_categories, n_features)

    def log_likelihood(self, samples: np.ndarray) -> np.ndarray:
        """Computes logP(X|θ)"""
        raise NotImplementedError

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
    gmm.fit(X)
