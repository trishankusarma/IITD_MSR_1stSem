import numpy as np
import itertools
import time
from collections import defaultdict, Counter
from svm import SupportVectorMachine

# ===============================
# One-vs-One Multi-class Wrapper
# ===============================
class OneVsOneSVM:
    def __init__(self, C=1.0, gamma=0.001, kernel='gaussian'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.models = {}
        self.pairs = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.pairs = list(itertools.combinations(self.classes_, 2))
        print(f"Training {len(self.pairs)} One-vs-One classifiers...")

        # ----------------------------------
        # Cache Gaussian kernel for all data
        # ----------------------------------
        if self.kernel == 'gaussian':
            K_full = np.exp(
                -self.gamma * (
                    np.sum(X**2, axis=1).reshape(-1, 1)
                    + np.sum(X**2, axis=1)
                    - 2 * np.dot(X, X.T)
                )
            )
        else:
            K_full = None

        # Train each binary classifier
        for (ci, cj) in self.pairs:
            mask = np.logical_or(y == ci, y == cj)
            X_pair, y_pair = X[mask], y[mask]
            y_pair = np.where(y_pair == ci, 1, -1)

            # Slice cached kernel for this pair
            if K_full is not None:
                idx = np.where(mask)[0]
                K_pair = K_full[np.ix_(idx, idx)]
            else:
                K_pair = None

            model = SupportVectorMachine()
            model.fit(X_pair, y_pair, kernel=self.kernel, C=self.C, gamma=self.gamma, K_precomputed=K_pair)
            self.models[(ci, cj)] = model

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))

        for (ci, cj), model in self.models.items():
            pred = model.predict(X)
            votes[:, np.where(self.classes_ == ci)[0][0]] += (pred > 0).astype(int)
            votes[:, np.where(self.classes_ == cj)[0][0]] += (pred < 0).astype(int)

        return self.classes_[np.argmax(votes, axis=1)]