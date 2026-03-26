import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class NaiveBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        N = len(y)

        # update pi
        Nk = np.asarray(np.unique(y, return_counts=True))[1, :]
        self.pi = Nk / N

        # update phi
        labels = np.unique(y)
        n_features = X.shape[1]
        self.phi = []
        for k in labels:
            Xk = X[y == k]
            for l in range(n_features):
                vals, counts = np.unique(Xk[:, l], return_counts=True)

                # total samples in class k
                Nk = len(Xk)

                # likelihood con smoothing
                alpha = 1
                prob = (counts + alpha) / (Nk + alpha * len(vals))

                # optional: creare array con lunghezza numero categorie totali
                # qui assumi che vals siano 0..n-1
                phi_l = np.zeros(int(np.max(X[:, l]) + 1))
                phi_l[vals.astype(int)] = prob

                self.phi.append(phi_l)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


if __name__ == "__main__":
    X, y = fetch_openml("mushroom", version=1, as_frame=False, return_X_y=True)

    X_train, X_test, y_train, y_test = [
        np.asarray(i) for i in train_test_split(X, y, test_size=0.2)
    ]

    imputer = SimpleImputer(strategy="most_frequent")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    feature_encoder = OrdinalEncoder()
    X_train = feature_encoder.fit_transform(X_train).astype(int)
    X_test = feature_encoder.transform(X_test).astype(int)

    label_encoder = LabelEncoder()
    y_train = np.asarray(label_encoder.fit_transform(y_train)).astype(int)
    y_test = label_encoder.transform(y_test).astype(int)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    y_pred_train = nb.predict(X_train)
    y_pred_test = nb.predict(X_test)

    print(f"train accuracy: {accuracy_score(y_train, y_pred_train):.2f}")
    print(f"test accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
