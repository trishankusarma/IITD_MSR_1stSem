import numpy as np
from cvxopt import matrix, solvers
# ===============================
# Support Vector Machine
# ===============================
class SupportVectorMachine:
    def __init__(self):
        self.w = None
        self.bias = None
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.kernel = None
        self.gamma = None
        self.C = None
        self.support_indices = None

    def _linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def _gaussian_kernel(self, X1, X2, gamma):
        X1 = np.array(X1)
        X2 = np.array(X2)
        if X1.ndim == 1:
            X1 = X1[np.newaxis, :]
        if X2.ndim == 1:
            X2 = X2[np.newaxis, :]
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dists)

    def fit(self, X, y, kernel='linear', C=1.0, gamma=0.001, K_precomputed=None):
        """
        Train the SVM using QP (CVXOPT)
        """
        X = np.array(X)
        y = np.array(y)
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        n_samples, n_features = X.shape

        # ----------------------------------
        # Kernel matrix (cached if provided)
        # ----------------------------------
        if K_precomputed is not None:
            K = K_precomputed
        elif kernel == 'linear':
            K = self._linear_kernel(X, X)
        elif kernel == 'gaussian':
            K = self._gaussian_kernel(X, X, gamma)
        else:
            raise ValueError("Unknown kernel")

        # Quadratic programming formulation
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((n_samples, 1)))
        A = matrix(y.astype(float), (1, n_samples))
        bias = matrix(0.0)

        if C is None or C == np.inf:
            G = matrix(-np.eye(n_samples))
            h = matrix(np.zeros(n_samples))
        else:
            G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))

        # ----------------------------------
        # Relax CVXOPT tolerances for speed
        # ----------------------------------
        solvers.options['show_progress'] = False
        solvers.options['abstol'] = 1e-3
        solvers.options['reltol'] = 1e-3
        solvers.options['feastol'] = 1e-3

        # Solve QP
        solution = solvers.qp(P, q, G, h, A, bias)
        alphas = np.ravel(solution['x'])

        # Support vectors
        sv_mask = alphas > 1e-5
        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.support_indices = np.where(sv_mask)[0]   # <-- store indices relative to training set

        # Compute bias term (bias)
        if kernel == 'linear':
            self.w = np.sum(
                (self.alphas * self.support_vector_labels)[:, None] * self.support_vectors,
                axis=0
            )
            self.bias = np.mean(
                self.support_vector_labels - np.dot(self.support_vectors, self.w)
            )
        else:
            self.w = None
            K_sv = K[np.ix_(sv_mask, sv_mask)]
            self.bias = np.mean(
                self.support_vector_labels -
                np.sum((self.alphas * self.support_vector_labels)[:, None] * K_sv, axis=0)
            )

    def project(self, X):
        X = np.array(X)
        if self.kernel == 'linear':
            return np.dot(X, self.w) + self.bias
        else:
            K = self._gaussian_kernel(X, self.support_vectors, self.gamma)
            return np.sum(
                (self.alphas * self.support_vector_labels) * K,
                axis=1
            ) + self.bias

    def predict(self, X):
        return np.sign(self.project(X))