import numpy as np
import matplotlib.pyplot as plt

class ModelClass:
    def __init__(self):
        pass
    def normalizeData(self, dfX, dfY): 
        
        print("Normalizing the given data into N(0,1)")
        X = (dfX - dfX.mean(axis=0)) / dfX.std(axis=0)
        Y = np.asarray([0 if y == "Alaska" else 1 for y in dfY])
        
        self.num_points ,self.num_features = X.shape
        
        self.X = X
        self.Y = Y
        
        print("Shape of X is :", self.X.shape)
        print("Shape of Y is :", self.Y.shape)
        
        return self.X, self.Y
    
    def getParameters(self):
        # phi0, phi1, mu0, mu1, sharedCovΣ
        # PriorProb(y = 'Canada')
        phi1 = np.mean(self.Y)
        print("Prior prob of P(y='Canada') :", phi1)
        # PriorProb(y = 'Alaska')
        phi0 = 1 - phi1
        print("Prior prob of P(y='Alaska') :", phi0)
        
        # get all rows in X with Y == 1 and get their means // Canada
        mu1 = np.mean( self.X[self.Y == 1], axis = 0)
        print("mean of data with (Y == 1) :: mu1 : ", mu1)
        
        # get all rows in X with  Y == 0 and get their means // Alaska
        mu0 = np.mean( self.X[self.Y == 0], axis = 0)
        print("mean of data with (Y == 0) :: mu0 : ", mu0)
        
        # get the sharedCovΣ
        sharedCovΣ = np.zeros((self.num_features, self.num_features))
        
        for data_idx in range(self.num_points):
            
            dataPoint = self.X[data_idx].reshape(-1, 1)
            mu = mu0 if self.Y[data_idx] == 0 else mu1
            diff = dataPoint - mu.reshape(-1, 1)
            sharedCovΣ += diff @ diff.T
        sharedCovΣ /= self.num_points
        print("sharedCovΣ: ", sharedCovΣ)
        
        # save the parameters into the model so that we can re-use them
        self.phi1 = phi1
        self.phi0 = phi0
        self.mu1 = mu1
        self.mu0 = mu0
        self.sharedCovΣ = sharedCovΣ
        
        return phi0, phi1, mu0, mu1, sharedCovΣ
    
    def plotTrainingData(self, ax=None):
        if ax is None:  # if no axes passed, create new one
            fig, ax = plt.subplots(figsize=(8,6))

        # Alaska points (Y=0)
        ax.scatter(self.X[self.Y==0,0], self.X[self.Y==0,1], 
                   c='blue', marker='o', label="Alaska (y=0)", alpha=0.7)

        # Canada points (Y=1)
        ax.scatter(self.X[self.Y==1,0], self.X[self.Y==1,1], 
                   c='red', marker='x', label="Canada (y=1)", alpha=0.7)

        ax.set_xlabel("X1 (normalized growth ring in fresh water)")
        ax.set_ylabel("X2 (normalized growth ring in marine water)")
        ax.set_title("Training Data: Alaska vs Canada Salmons")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        return ax
    
    def getParametersForDecisionBoundaryForLinearGDA(self):
        sharedCovΣInv = np.linalg.pinv(self.sharedCovΣ)
        self.W = sharedCovΣInv @ (self.mu1 - self.mu0)
        
        term1 = (self.mu0.T @ sharedCovΣInv @ self.mu0 - self.mu1.T @ sharedCovΣInv @ self.mu1)/2
        term2 = np.log(self.phi1 / self.phi0)
        self.b = term1 + term2

        print("value of W: ", self.W, " And value of b ", self.b)
        return self.W, self.b
        
    def plotLinearGDA(self):
        fig, ax = plt.subplots(figsize=(8,6))
        self.plotTrainingData(ax=ax)

        # Decision boundary
        x_vals = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 200)
        y_vals = -(self.W[0]/self.W[1]) * x_vals - self.b/self.W[1]
        ax.plot(x_vals, y_vals, 'k-', linewidth=2, label="Decision boundary")
        ax.legend()

        return ax

    
    def getActualCovarianceMatrices(self):
        
        # get the rows for Y == 0 and Y == 1 separately
        X0 = self.X[self.Y == 0]
        X1 = self.X[self.Y == 1]

        # covariance for class 0
        diffs0 = X0 - self.mu0   # shape (m0, n)
        self.Σ0 = (diffs0.T @ diffs0) / X0.shape[0]
        
        print("Covariance Matrix Σ0 is : ", self.Σ0)

        # covariance for class 1
        diffs1 = X1 - self.mu1
        self.Σ1 = (diffs1.T @ diffs1) / X1.shape[0]
        
        print("Covariance Matrix Σ1 is : ", self.Σ1)
        
        return self.Σ0, self.Σ1
    
    def getQuadraticGDAParams(self):
        
        Σ0Inv = np.linalg.pinv(self.Σ0)
        Σ1Inv = np.linalg.pinv(self.Σ1)
        
        self.P = (Σ0Inv - Σ1Inv)/2
        self.Q = self.mu1.T @ Σ1Inv - self.mu0.T@Σ0Inv 
        
        term1 = (self.mu0.T @ Σ0Inv @  self.mu0 - self.mu1.T @ Σ1Inv @  self.mu1 )/2
        term2 = np.log(np.linalg.det(self.Σ1)/np.linalg.det(self.Σ0))
        self.R = term1 + term2
        
        print("Value of P : ", self.P)
        print("Value of Q : ", self.Q)
        print("Value of R : ", self.R)
        
        return self.P, self.Q, self.R

    def plot_quadratic_boundary(self, addLinearBoundary = False):
        """
        Plot decision boundary given x^T P x + Q^T x + R = 0
        P: (2,2) numpy array
        Q: (2,) numpy array
        R: scalar
        X: training data (to get plot limits)
        """
        x_min, x_max = self.X[:,0].min() - 1, self.X[:,0].max() + 1
        y_min, y_max = self.X[:,1].min() - 1, self.X[:,1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                             np.linspace(y_min, y_max, 400))
        grid = np.c_[xx.ravel(), yy.ravel()]
        vals = np.einsum('ij,nj,ni->n', self.P, grid, grid) + grid @ self.Q + self.R
        vals = vals.reshape(xx.shape)

        if not addLinearBoundary:
            fig, ax = plt.subplots(figsize=(8,6))
            self.plotTrainingData(ax=ax)
        else:
            ax = self.plotLinearGDA()   # reuse linear plot

        ax.contour(xx, yy, vals, levels=[0], colors='k', linewidths=2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Quadratic Decision Boundary (x^T P x + Q^T x + R = 0)")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.show()
        return ax

        