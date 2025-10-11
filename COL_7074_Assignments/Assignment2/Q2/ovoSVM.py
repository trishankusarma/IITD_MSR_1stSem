import numpy as np
import itertools
import time
from collections import defaultdict, Counter
from svm import SupportVectorMachine

class OneVsOneSVM:
    """
    One-vs-One Multi-Class SVM using CVXOPT-based binary SVMs
    """

    def __init__(self, C=1.0, gamma=0.001):
        self.C = C
        self.gamma = gamma
        self.models = {}
        self.pairs = []
        self.classes_ = None

    def fit(self, X, y, kernel='gaussian', C=1.0):
        """
        Train a binary SVM for every pair of classes
        """
        self.C = C
        self.classes_ = np.unique(y)
        self.pairs = list(itertools.combinations(self.classes_, 2))
        print(f"Training {len(self.pairs)} One-vs-One classifiers...")
        start = time.time()

        for (ci, cj) in self.pairs:
            print(f"\n Training classifier for ({ci} vs {cj})")
            mask = np.logical_or(y == ci, y == cj)
            X_pair, y_pair = X[mask], y[mask]
            y_pair = np.where(y_pair == ci, 1, -1)

            svm = SupportVectorMachine()
            svm.fit(X_pair, y_pair, kernel="gaussian", C=C, gamma=self.gamma)
            self.models[(ci, cj)] = svm

        end = time.time()
        print(f"\nOne-vs-One training completed in {end - start:.2f}s")

    def predict(self, X):
        """
        Predict using majority voting across all pairwise classifiers
        """
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes_)))
        decision_scores = np.zeros_like(votes)

        for (ci, cj), svm in self.models.items():
            # compute decision function values for each test sample
            K = svm._gaussian_kernel(X, svm.support_vectors)
            decision_values = np.sum(svm.alphas * svm.support_labels * K, axis=1) + svm.bias

            # assign votes
            for idx, val in enumerate(decision_values):
                if val > 0:
                    votes[idx, np.where(self.classes_ == ci)[0][0]] += 1
                    decision_scores[idx, np.where(self.classes_ == ci)[0][0]] += val
                else:
                    votes[idx, np.where(self.classes_ == cj)[0][0]] += 1
                    decision_scores[idx, np.where(self.classes_ == cj)[0][0]] += -val

        # handle ties — pick label with highest total decision score
        preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            max_votes = np.max(votes[i])
            candidates = np.where(votes[i] == max_votes)[0]
            if len(candidates) == 1:
                preds[i] = self.classes_[candidates[0]]
            else:
                # tie-break using decision scores
                best_idx = candidates[np.argmax(decision_scores[i, candidates])]
                preds[i] = self.classes_[best_idx]

        return preds
