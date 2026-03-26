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
        self.pi = np.asarray(np.unique(y, return_counts=True))[1, :] / N

        values = [np.unique(X, axis=0) for i in range(X.shape[1])]
        print(values)
        for v in values:
            counts = np.asarray(np.unique(v, return_counts=True))[1, :]
            print(counts)

        # for v in values:
        #     print(v)
        # self.phi = [np.zeros_like(values[i] for i in values)]
        # for v in self.phi:
        #     print(v)

        #
        # print(self.pi)
        # print(self.phi)
        #
        # for label in labels:
        #     Nk = y[y == label]
        #     self.pi[label] = Nk / N

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

    # y_pred_train = nb.predict(X_train)
    # y_pred_test = nb.predict(X_test)
    #
    # print(f"train accuracy: {accuracy_score(y_train, y_pred_train):.2f}")
    # print(f"test accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
