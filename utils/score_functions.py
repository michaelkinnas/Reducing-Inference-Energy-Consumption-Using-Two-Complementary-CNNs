from torch import softmax
from torch import max as tmax
from math import log


def max_probability(logits):
    probs = softmax(logits, dim=1)
    max_prob = tmax(probs).item()
    return max_prob


def difference(logits):
    probs = softmax(logits, dim=1)
    probs = probs[0].tolist()
    first = max(probs)
    probs[probs.index(max(probs))] = 0
    second = max(probs)
    difference = first - second
    return difference


def entropy(logits):
    probs = softmax(logits, dim=1)
    entropy = -sum([x.item() * log(x.item()) for x in probs[0]])
    return entropy