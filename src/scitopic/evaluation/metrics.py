import itertools
from collections import Counter

import numpy as np
import topmost


def calculate_topic_diversity(top_words, top_n=10):
    # Gather top n words for each topic
    top_words = [set([word for word in top_words[i]]) for i in range(len(top_words))]
    # Calculate pairwise intersections
    unique_pairs = [(top_words[i], top_words[j]) for i in range(len(top_words)) for j in range(i+1, len(top_words))]
    total_intersections = sum(len(set_a.intersection(set_b)) for set_a, set_b in unique_pairs)
    total_possible = len(unique_pairs) * top_n
    return 1 - total_intersections / total_possible

def calculate_topic_coherence_topmost(dataset, top_words):
    return topmost.evaluations.compute_topic_coherence(dataset.train_texts, dataset.vocab, top_words)

def calculate_word_statistics(documents, window_size=None):
    word_counter = Counter()
    cooccurrence_counter = Counter()
    total_words = 0

    for doc in documents:
        words = list(doc)
        total_words += len(words)
        word_counter.update(words)

        if window_size is None:
            word_pairs = itertools.combinations(set(words), 2)
        else:
            word_pairs = [
                (words[i], words[j])
                for i in range(len(words))
                for j in range(i + 1, min(i + window_size, len(words)))
            ]
        cooccurrence_counter.update(word_pairs)

    return word_counter, cooccurrence_counter, total_words

def calculate_npmi(word_counter, cooccurrence_counter, total_words):
    npmi_scores = {}
    total_pairs = sum(cooccurrence_counter.values())

    for (w1, w2), cooccurrence_count in cooccurrence_counter.items():
        p_w1 = word_counter[w1] / total_words
        p_w2 = word_counter[w2] / total_words
        p_w1_w2 = cooccurrence_count / total_pairs

        if p_w1_w2 > 0:
            pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
            npmi = pmi / -np.log(p_w1_w2)
            npmi_scores[(w1, w2)] = npmi

    return npmi_scores

def calculate_topic_coherence_npmi(top_words, npmi_scores):
    pair_scores = []
    for w1, w2 in itertools.combinations(top_words, 2):
        if (w1, w2) in npmi_scores:
            pair_scores.append(npmi_scores[(w1, w2)])
        elif (w2, w1) in npmi_scores:
            pair_scores.append(npmi_scores[(w2, w1)])

    if pair_scores:
        return np.mean(pair_scores)
    else:
        return 0.0
