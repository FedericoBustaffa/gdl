import argparse
import re

import numpy as np

from models import MarkovChain


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
