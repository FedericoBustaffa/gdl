import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import GaussianMixtureModel

if __name__ == "__main__":
    np.random.seed(9951)

    df = pd.read_csv("midterm1/train.csv")
    X = df.to_numpy()

    gmms = [GaussianMixtureModel(k, X.shape[1]) for k in range(1, 5)]
    for gmm in tqdm(gmms, desc="validation", ncols=80):
        gmm.fit(X)

    for gmm in gmms:
        print(f"number of clusters: {gmm.n_categories}: ", end="")
        print(f"log likelihood: {gmm.log_likelihood(X):.4f}, ", end="")
        print(f"BIC: {gmm.bic(X)}")

    plt.figure(dpi=100)
    plt.title("Log-Likelihood Curve")
    for gmm in gmms:
        plt.plot(gmm.log_likelihood_history, label=f"K = {gmm.n_categories}")
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(dpi=100)
    plt.title("BIC Curve")
    for gmm in gmms:
        plt.plot(gmm.bic_history, label=f"K = {gmm.n_categories}")
    plt.xlabel("Iterations")
    plt.ylabel("BIC")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
