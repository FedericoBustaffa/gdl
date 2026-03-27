import numpy as np
from tqdm import trange


class MarkovChain:
    def __init__(self, n_states: int, transition_matrix: np.ndarray) -> None:
        self.states = [i for i in range(n_states)]
        self.transition_matrix = transition_matrix

    def generate(
        self, initial_state_dist: np.ndarray, max_iter: int = 200
    ) -> np.ndarray:
        # initial state probabilities must sum to 1
        if np.sum(initial_state_dist) != 1.0:
            raise ValueError()

        # output sequence
        state = np.random.choice(self.states, p=initial_state_dist)
        sequence = [state]

        # generate the sequence of states
        for _ in range(max_iter):
            next_state = np.random.choice(self.states, p=self.transition_matrix[state])
            sequence.append(next_state)
            state = next_state

        return np.array(sequence)

    def fit(self, X: list) -> None:
        counters = np.zeros_like(self.transition_matrix)
        for i in trange(len(X), ncols=80, desc="training"):
            for j in range(1, len(X[i])):
                s0 = X[i][j - 1]
                s1 = X[i][j]

                counters[s0][s1] += 1

        rows_totals = counters.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            counters, rows_totals, where=rows_totals > 0, out=np.zeros_like(counters)
        )
