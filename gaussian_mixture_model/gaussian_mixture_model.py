import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gaussian_pdf(mu: np.ndarray, sigma: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X - mu
    exponent = -(diff**2) / (2 * sigma**2)
    pdf = np.exp(exponent) / (np.sqrt(2 * np.pi) * sigma)

    return pdf


class GaussianMixtureModel:
    def __init__(self, n_categories: int, n_features: int):
        self.n_categories = n_categories
        self.n_features = n_features

        # every cluster equiprobable
        self._pi = np.ones(shape=n_categories) * (1 / n_categories)

        # means all at zero
        self._mu = np.random.randn(self.n_categories, self.n_features)

        # variances all at one to prevent initial division by zero
        self._sigma = np.ones((self.n_categories, self.n_features))

    def log_likelihood(self, samples) -> np.ndarray:
        """Computes logP(X|θ)"""

        probabilities = np.zeros(shape=(self.n_categories, samples.shape[0]))

        for m in range(self.n_categories):
            pdf = gaussian_pdf(self._mu[m], self._sigma[m], samples)
            probabilities[m] = np.prod(pdf, axis=1)

        mixture = np.sum(self._pi[:, None] * probabilities, axis=0)

        return np.sum(np.log(mixture))

    def fit(self, samples: np.ndarray, n_iter=100):

        N_samples, D = samples.shape

        prev_log_likelihood = self.log_likelihood(samples)

        for i in range(n_iter):
            # E STEP
            probabilities = np.zeros((self.n_categories, N_samples))

            for m in range(self.n_categories):
                pdf = gaussian_pdf(self._mu[m], self._sigma[m], samples)
                probabilities[m] = np.prod(pdf, axis=1) * self._pi[m]

            denom = np.sum(probabilities, axis=0)
            r = probabilities / denom

            # M STEP
            Nk = np.sum(r, axis=1)

            for m in range(self.n_categories):
                weights = r[m][:, None]

                self._pi[m] = Nk[m] / N_samples

                self._mu[m] = np.sum(weights * samples, axis=0) / Nk[m]

                diff = samples - self._mu[m]

                self._sigma[m] = np.sqrt(np.sum(weights * diff**2, axis=0) / Nk[m])

            print(f"log likelihood: {self.log_likelihood(samples):.4f}")
            curr_log_likelihood = self.log_likelihood(samples)
            if abs(curr_log_likelihood - prev_log_likelihood) < 1e-4:
                print(f"converged in {i} iterations")
                return

            prev_log_likelihood = curr_log_likelihood

    def bic(self, samples: np.ndarray) -> float:
        N, D = samples.shape
        K = self.n_categories

        log_likelihood = self.log_likelihood(samples)

        # numero di parametri
        p = 2 * K * D + (K - 1)

        return log_likelihood - 0.5 * p * np.log(N)


if __name__ == "__main__":
    np.random.seed(0)

    df = pd.read_csv("gaussian_mixture_model/train.csv")
    X = df.to_numpy()

    gmm = GaussianMixtureModel(4, X.shape[1])
    l = gmm.log_likelihood(X)
    print(f"log likelihood: {l}")

    gmm.fit(X)

    print(f"BIC: {gmm.bic(X)}")
