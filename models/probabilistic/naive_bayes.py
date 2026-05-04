import numpy as np


class NaiveBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        N = len(y)
        self.labels = np.unique(y)

        # prior
        Nk = np.asarray(np.unique(y, return_counts=True))[1]
        self.pi = Nk / N

        n_features = X.shape[1]
        self.phi = {}

        for k in self.labels:
            Xk = X[y == k]
            Nk = len(Xk)

            self.phi[k] = []

            for l in range(n_features):
                vals, counts = np.unique(Xk[:, l], return_counts=True)

                alpha = 1
                prob = (counts + alpha) / (Nk + alpha * len(vals))

                phi_l = np.zeros(int(np.max(X[:, l]) + 1))
                phi_l[vals.astype(int)] = prob

                self.phi[k].append(phi_l)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []

        for x in X:
            scores = []

            for idx, k in enumerate(self.labels):
                score = np.log(self.pi[idx])

                for l, val in enumerate(x):
                    prob = self.phi[k][l][int(val)]
                    if prob > 0:
                        score += np.log(prob)
                    else:
                        score += -1e9

                scores.append(score)

            predictions.append(self.labels[np.argmax(scores)])

        return np.array(predictions)
