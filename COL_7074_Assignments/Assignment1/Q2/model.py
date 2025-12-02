import numpy as np
import time

class modelClass:
    def __init__(self, params, gaussianDetails, total_data_points, train_test_spit_ratio):

        print("Model for getting sampling data with gaussian noise")

        self.params = params
        self.gaussianDetails = gaussianDetails
        self.total_data_points = total_data_points
        self.train_test_spit_ratio = train_test_spit_ratio
        self.num_features = 2

        # Intercept term (x0 = 1)
        X0 = np.ones(total_data_points)

        X1 = np.random.normal(
            loc = self.gaussianDetails["x1"]["mean"],
            scale = np.sqrt( self.gaussianDetails["x1"]["variance"] ),
            size = self.total_data_points
            ) # N(3, 4)

        X2 = np.random.normal(
            loc = self.gaussianDetails["x2"]["mean"],
            scale = np.sqrt( self.gaussianDetails["x2"]["variance"] ),
            size = self.total_data_points
            ) # N(-1, 4)

        # Stack features into design matrix X
        self.X = np.vstack((X0, X1, X2))
        print("Shape of X :",self.X.shape)

        Y_Noise = np.random.normal(
            loc = self.gaussianDetails["y_noice"]["mean"],
            scale = np.sqrt( self.gaussianDetails["y_noice"]["variance"] ),
            size = self.total_data_points
            ) # N(0, 2)

        theta = np.array([self.params["theta0"], self.params["theta1"], self.params["theta2"]])

        # Y = theta0 + theta1*x1 + theta2*x2 + N(0,2)
        self.Y = (np.dot(self.X.T, theta) + Y_Noise).reshape((1, total_data_points))
        print("Shape of Y :",self.Y.shape)
        
        # shuffling before spliting the data
        print("Shuffling before spling the data")
        indices = np.arange(self.total_data_points)
        np.random.shuffle(indices)
        self.X, self.Y = self.X[:,indices], self.Y[:,indices]

        self.train_size = int(self.train_test_spit_ratio * self.total_data_points)
        
        self.train_data = self.X[:, :self.train_size], self.Y[:, :self.train_size]
        self.test_data = self.X[:, self.train_size:], self.Y[:, self.train_size:]
        
        train_x, train_y = self.train_data
        test_x, test_y = self.test_data
        print("Training set X size : ", train_x.shape)
        print("Training set Y size : ", train_y.shape)
        print("Testing set X size : ", test_x.shape)
        print("Testing set Y size : ", test_y.shape)
    
    def initializeParams(self):
        theta = np.zeros((self.num_features+1, 1))
        print("Shape of theta : ", theta.shape)
        return theta
    
    def isConvergeCriteriaMet(self, epoc, total_epocs, loss_history, prevTheta, currTheta, tolerance, window = 5):
        # Condition 1: Max epochs
        if epoc >= total_epocs:
            return True

        # Condition 2: Moving average stabilization
        if len(loss_history) > window:
            recent_avg = np.mean(loss_history[-window:])
            prev_avg   = np.mean(loss_history[-(2*window):-window])
            if abs(recent_avg - prev_avg) < tolerance:
                return True

        # Condition 3: Parameter stabilization (L2 norm)
        if np.linalg.norm(currTheta - prevTheta) < tolerance:
            return True

        return False

    # GET TOTAL ERROR
    def get_total_error(self, theta):
        train_x, train_y = self.train_data
        # return J(theta) = 1/(2m) sum( np.square( <theta.transpose, X> - Y ))
        return np.sum(np.square(np.dot(theta.T, train_x) - train_y)) / (2 * self.train_size)

    # GET GRADIENT ON THETA
    def get_gradient(self, X, Y, theta, batch_size):
        # return grad = 1/m (<X ,(<theta.transpose, X> - Y).Transpose ) >
        loss = np.dot(theta.T, X) - Y
        return np.dot(X, loss.T) / batch_size
    
    def runBatchGradientDescent(self, theta, batch_size=1, learning_rate=0.01, total_epocs=5000, tolerance=1e-9, window = 5):
        print("Running Batch Gradient Descent for batch_size:", batch_size, " and learning_rate:", learning_rate)

        # initialize theta
        epocs = 0
        loss = []
        
        # training_data_set_size
        m = self.train_size
        
        startTime = time.time()
        timeTaken = 0
        thetaHistory = []

        while epocs < total_epocs:
            # Shuffle indices only (don't destroy original X, Y)
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuff, Y_shuff = self.X[:, indices], self.Y[:, indices]

            prevTheta = theta.copy()

            # Mini-batch updates
            for i in range(0, m, batch_size):
                currX = X_shuff[:, i:i+batch_size]
                currY = Y_shuff[:, i:i+batch_size]

                grad = self.get_gradient(currX, currY, theta, batch_size)
                theta = theta - learning_rate * grad

            # Compute loss on full dataset after epoch
            error = self.get_total_error(theta)
            thetaHistory.append(theta)
            loss.append(error)

            # Convergence check
            if epocs > 0:
                isConvergeCriteriaTrue = self.isConvergeCriteriaMet(
                    epocs, total_epocs, loss, prevTheta, theta, tolerance, window = window
                )
            else:
                isConvergeCriteriaTrue = False
                
            isFullBatch = batch_size == self.train_size
            
            logCriteria = 8 if not isFullBatch else 500

            if epocs % logCriteria == 0 or isConvergeCriteriaTrue:
                print(f"Epoch {epocs}, Loss: {loss[-1]}")
                print(f"Theta : {theta}")
                endTime = time.time()
                print(f"Execution time for prev epocs: {endTime - startTime:.4f} seconds")
                timeTaken += endTime - startTime
                startTime = endTime

            if isConvergeCriteriaTrue:
                break

            epocs += 1

        return theta, loss, epocs, timeTaken, thetaHistory
