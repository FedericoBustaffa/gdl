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


def generate_sentence(mc, idx_to_word, word_to_idx, max_len=30):
    current = word_to_idx["<START>"]
    sentence = []

    for _ in range(max_len):
        probs = mc.transition_matrix[current]

        # se per qualche motivo è tutto zero → stop
        if probs.sum() == 0:
            break

        next_state = np.random.choice(len(probs), p=probs)
        next_word = idx_to_word[next_state]

        if next_word == "<END>":
            break

        sentence.append(next_word)
        current = next_state

    return " ".join(sentence)


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="filepath of the dataset")
    args = parser.parse_args()

    book = open(args.filepath).read().lower()
    book = re.sub(r"[^a-zàèéìòù\s\.]", "", book)

    sentences = [
        ["<START>"] + s.split() + ["<END>"] for s in book.split(".") if s.strip()
    ]

    vocabulary = list(set(w for s in sentences for w in s))
    n_states = len(vocabulary)

    word_to_idx = {w: i for i, w in enumerate(vocabulary)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    sequences = [np.array([word_to_idx[w] for w in sentence]) for sentence in sentences]

    transition_matrix = np.zeros((n_states, n_states))
    mc = MarkovChain(n_states=n_states, transition_matrix=transition_matrix)

    mc.fit(sequences)

    print(generate_sentence(mc, idx_to_word, word_to_idx, 30))
