import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gaussian(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    diff = x - mu
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
        self._mu = np.random.random(size=(self.n_categories, self.n_features))

        # variances all at one to prevent initial division by zero
        self._sigma = np.ones((self.n_categories, self.n_features))

    def log_likelihood(self, samples: np.ndarray) -> np.ndarray:
        """Computes logP(X|θ)"""

        probabilities = np.zeros(shape=(samples.shape[0]))
        for i, x in enumerate(samples):
            weighted = np.zeros(self.n_categories)

            for m in range(self.n_categories):
                pdfs = gaussian(self._mu[m], self._sigma[m], x)
                proba = np.prod(pdfs)
                weighted[m] = proba * self._pi[m]

            probabilities[i] = np.log(np.sum(weighted))

        return np.asarray(np.sum(probabilities))

    def fit(self, samples: np.ndarray):
        """Fits the GMM parameters using the EM algorithm."""

        n_samples = samples.shape[0]

        # for convergence check
        self.log_likelihood_history = [self.log_likelihood(samples)]

        for _ in range(200):
            # E-step
            responsibilities = np.zeros(shape=(n_samples, self.n_categories))

            for i, x in enumerate(samples):
                weighted = np.zeros(shape=(self.n_categories))
                for m in range(self.n_categories):
                    # for each sample compute the probability of each gaussian
                    pdfs = gaussian(self._mu[m], self._sigma[m], x)

                    # exploit conditional independence
                    proba = np.prod(pdfs)

                    weighted[m] = proba * self._pi[m]

                responsibilities[i] = weighted / np.sum(weighted)

            # M-step
            Nm = np.sum(responsibilities, axis=0)
            for m in range(self.n_categories):
                # update pi
                self._pi[m] = Nm[m] / n_samples

                # update mu
                mu_num = np.zeros(self.n_features)
                for i in range(n_samples):
                    mu_num += responsibilities[i, m] * samples[i]
                self._mu[m] = mu_num / Nm[m]

                # update sigma
                sigma_num = np.zeros(self.n_features)
                for i in range(n_samples):
                    diff = samples[i] - self._mu[m]
                    sigma_num += responsibilities[i, m] * (diff**2)
                self._sigma[m] = np.sqrt(sigma_num / Nm[m])

            curr_log_likelihood = self.log_likelihood(samples)

            # convergence check
            if abs(curr_log_likelihood - self.log_likelihood_history[-1]) < 1e-4:
                return

            self.log_likelihood_history.append(curr_log_likelihood)

    def bic(self, samples: np.ndarray) -> float:
        """Computes the BIC score."""
        n, d = samples.shape
        k = self.n_categories

        log_likelihood = self.log_likelihood(samples)

        # number of parameters
        p = 2 * k * d + (k - 1)

        return log_likelihood - 0.5 * p * np.log(n)


if __name__ == "__main__":
    np.random.seed(0)

    df = pd.read_csv("midterm1/train.csv")
    X = df.to_numpy()

    gmm = GaussianMixtureModel(4, X.shape[1])
    log_likelihood = gmm.log_likelihood(X)
    print(f"log likelihood: {log_likelihood:.4f}")

    gmm.fit(X)
    print(f"BIC: {gmm.bic(X)}")

    plt.plot(gmm.log_likelihood_history)
    plt.show()
