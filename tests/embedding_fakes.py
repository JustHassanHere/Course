"""Test doubles for the embedding model, so no test downloads it."""

import re

import numpy as np

from classbuild.rag.embeddings import EMBEDDING_DIM

KEYWORDS = [
    "python", "list", "lists", "tuple", "tuples", "mutable", "immutable", "dynamic", "dynamically",
    "typed", "variable", "variables", "function", "functions", "loop", "loops", "string", "strings",
    "dictionary", "kubernetes", "pod", "pods", "cluster", "container", "indentation",
]


class FakeEncoder:
    """Builds 384-number vectors by counting a fixed list of keywords, so texts on the
    same topic really are similar. Texts without any keyword share one 'other' direction.
    Records each encode() call."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        matrix = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z]+", text.lower())
            for index, keyword in enumerate(KEYWORDS):
                matrix[row, index] = words.count(keyword)
            if not matrix[row].any():
                matrix[row, EMBEDDING_DIM - 1] = 1.0
        return matrix

    @property
    def texts_encoded(self):
        return sum(len(call) for call in self.calls)


class BrokenEncoder:
    """Returns a chosen bad output or raises a chosen error."""

    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error

    def encode(self, texts, **kwargs):
        if self.error is not None:
            raise self.error
        return self.output
