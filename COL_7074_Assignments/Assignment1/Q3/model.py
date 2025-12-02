import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelClass:
    def __init__(self):
        pass

    def initializeXYAndParams(self, dfX, dfY):
        self.num_points, self.num_features = dfX.shape
        X = dfX.to_numpy()
        Y = dfY.to_numpy().reshape(-1, 1)
        
        print("Shape of X : ", X.shape)
        print("Shape of Y : ", Y.shape)

        # normalize features
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
        # add intercept
        self.X = np.append(np.ones((X_norm.shape[0], 1)), X_norm, axis = 1)
        self.Y = Y

        # initialize theta
        THETA_AUG = np.zeros((self.num_features + 1, 1))

        print("X.shape after augmentation:", self.X.shape)
        print("Y.shape:", self.Y.shape)
        print("THETA_AUG.shape:", THETA_AUG.shape)

        return self.X, self.Y, THETA_AUG

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def getLoss(self, theta):
        hTheta = self.sigmoid(np.dot(self.X, theta))
        epsilon = 1e-15
        hTheta = np.clip(hTheta, epsilon, 1 - epsilon)
        loss = -np.sum(self.Y * np.log(hTheta) + (1 - self.Y) * np.log(1 - hTheta))
        return loss

    def calculateGrad(self, theta):
        # Gradient of negative log-likelihood
        hTheta = self.sigmoid(np.dot(self.X, theta))  # (m,1)
        error = self.Y - hTheta
        grad = self.X.T @ error    # (n,1)
        return -grad

    def calculateHessian(self, theta):
        hTheta = self.sigmoid(np.dot(self.X, theta)).flatten()
        r = hTheta * (1 - hTheta)
        # Efficient diagonal multiplication
        XtRX = self.X.T @ (r[:, np.newaxis] * self.X)
        return XtRX

    def optimizeLikelihoodFunction(self, theta, maxEpocs=100000, tolerance=1e-9, alpha=0.001):
        currLoss = self.getLoss(theta)
        print("Starting loss:", currLoss)

        for epoc in range(1, maxEpocs + 1):
            grad = self.calculateGrad(theta)
            H = self.calculateHessian(theta)

            delta = np.linalg.pinv(H) @ grad
            theta_new = theta - alpha * delta

            newLoss = self.getLoss(theta_new)

            if epoc % 1000 == 0 or np.linalg.norm(theta_new - theta) < tolerance:
                print(f"Epoch {epoc}, Loss: {newLoss:.6f}, Theta: {theta_new.T}")

            if np.linalg.norm(theta_new - theta) < tolerance:
                print(f"Converged :: Optimal Values of theta that we got are :: {theta}!")
                return theta_new

            theta = theta_new

        print("Reached maximum epochs.")
        return theta
    
    def plotDecisionBoundary(self, theta):
        # Scatter plot of training data
        pos = (self.Y.flatten() == 1)
        neg = (self.Y.flatten() == 0)

        plt.scatter(self.X[pos, 1], self.X[pos, 2], marker='o', label='Class 1')
        plt.scatter(self.X[neg, 1], self.X[neg, 2], marker='x', label='Class 0')

        # Decision boundary: theta0 + theta1*x1 + theta2*x2 = 0
        x1_vals = np.linspace(self.X[:,1].min(), self.X[:,1].max(), 100)
        x2_vals = -(theta[0] + theta[1] * x1_vals) / theta[2]

        plt.plot(x1_vals, x2_vals, 'g-', linewidth=2, label='Decision boundary')

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.title("Logistic Regression Decision Boundary")
        plt.show()
