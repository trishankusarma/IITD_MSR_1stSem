import numpy as np
import cvxopt

class SupportVectorMachine:
    """
    Binary Classifier using Support Vector Machine (SVM)
    Solved via Quadratic Programming using CVXOPT.
    Works for both linear and Gaussian (RBF) kernels.
    """

    def __init__(self):
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = 0
        self.kernel = None
        self.C = None
        self.gamma = None
        self.w = None
        self.support_indices = None

    # ----------------------------
    # Kernels
    # ----------------------------
    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def _gaussian_kernel(self, x1, x2):
        if x1.ndim == 1:
            x1 = x1[np.newaxis, :]
        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]
        
        # Efficiently compute squared distances without creating 3D arrays
        x1_sq = np.sum(x1 ** 2, axis=1).reshape(-1, 1)
        x2_sq = np.sum(x2 ** 2, axis=1).reshape(1, -1)
        sq_dists = x1_sq + x2_sq - 2 * np.dot(x1, x2.T)
    
        # Ensure no negative values due to numerical errors
        sq_dists = np.maximum(sq_dists, 0.0)
    
        return np.exp(-self.gamma * sq_dists)

    # ----------------------------
    # Fit
    # ----------------------------
    def fit(self, X, y, kernel='linear', C=1.0, gamma=0.001):
        """
        Train SVM model using CVXOPT solver.
        Args:
            X: np.array, shape (N, D)
            y: np.array, shape (N,)
            kernel: 'linear' or 'gaussian'
            C: Soft margin regularization parameter
            gamma: RBF kernel parameter (used only if kernel='gaussian')
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Convert labels only if necessary
        if np.any(np.isin(y, [0, 1])):
            y[y == 0] = -1

        n_samples, n_features = X.shape
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

        # Compute kernel matrix
        if kernel == 'linear':
            K = self._linear_kernel(X, X)
        elif kernel == 'gaussian':
            K = self._gaussian_kernel(X, X)
        else:
            raise ValueError("Unsupported kernel type: use 'linear' or 'gaussian'")

        # Quadratic Programming setup
        P = cvxopt.matrix(np.outer(y, y) * K + 1e-10 * np.eye(n_samples))
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0)

        G_std = np.diag(-np.ones(n_samples))
        h_std = np.zeros(n_samples)
        G_slack = np.diag(np.ones(n_samples))
        h_slack = np.ones(n_samples) * C

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.hstack((h_std, h_slack)))

        # Solve QP
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])

        # Support vectors
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.support_indices = np.where(sv)[0]   # <-- store indices relative to training set

        # Compute bias and weight
        if kernel == 'linear':
            self.w = np.sum(
                self.alphas[:, None] * self.support_labels[:, None] * self.support_vectors,
                axis=0,
            )
            self.bias = np.mean(
                self.support_labels - np.dot(self.support_vectors, self.w)
            )
        else:
            self.bias = np.mean([
                self.support_labels[i] - np.sum(
                    self.alphas * self.support_labels *
                    self._gaussian_kernel(self.support_vectors[i], self.support_vectors)
                )
                for i in range(len(self.alphas))
            ])
            self.bias = np.clip(self.bias, -10, 10)  # stability
        

    # ----------------------------
    # Predict
    # ----------------------------
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if self.kernel == 'linear':
            y_pred = np.dot(X, self.w) + self.bias
        else:
            K = self._gaussian_kernel(X, self.support_vectors)
            y_pred = np.sum(self.alphas * self.support_labels * K, axis=1) + self.bias

        return np.sign(y_pred)  # outputs +1 / -1