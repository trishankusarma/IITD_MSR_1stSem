import numpy as np
from IPython.display import display
import time
import matplotlib.pyplot as plt
import seaborn as sns

class ModelClass:
    def __init__(self):
        self.num_points = 0
        self.num_features = 0
        self.X = {}
        self.Y = {}

    def initializeXYAndParams(self, dfX, dfY):
        # dfX -> ( num_points, num_features )
        # dfY -> ( num_points, 1 )
        self.num_points, self.num_features = dfX.shape

        # CONVERTING X TO COLUMN DATA POINTS (num_features, num_examples)
        X = dfX.to_numpy().T   # shape: (n, m)

        # AUGMENTING A NEW ROW OF 1s
        X_AUG = np.vstack((np.ones((1, self.num_points)), X))
        print("X.shape after augmentation: ", X_AUG.shape)

        # MAKING Y A ROW VECTOR (1, m)
        Y = dfY.to_numpy().T   # shape: (1, m)
        print("Y.shape after transpose: ", Y.shape)

        # INITIALIZING PARAMETERS
        THETA = np.zeros((self.num_features, 1))
        BIAS = np.zeros((1, 1))
        print("THETA.shape : ", THETA.shape)
        print("BIAS.shape : ", BIAS.shape)

        # AUGMENTING THETA (bias + weights)
        THETA_AUG = np.vstack((BIAS, THETA))
        print("THETA.shape after augmenting : ", THETA_AUG.shape)
        
        self.X = X_AUG
        self.Y = Y

        return X_AUG, Y, THETA_AUG
    
    # GET TOTAL ERROR
    def get_total_error(self, theta):
      # return J(theta) = 1/(2m) sum( np.square( <theta.transpose, X> - Y ))
      return np.sum(np.square(np.dot(theta.T, self.X) - self.Y)) / (2 * self.num_points)
    
    # GET GRADIENT ON THETA
    def get_gradient(self, theta):
        # return grad = 1/m (<X ,(<theta.transpose, X> - Y).Transpose ) >
        loss = np.dot(theta.T, self.X) - self.Y
        return np.dot(self.X, loss.T) / self.num_points
    
    def isConvergeCriteriaMet(self,epocs, total_epocs, lossPrev, lossCurr, tolerance):
        return ((epocs+1) >= total_epocs) or (epocs > 1 and (lossPrev - lossCurr) < tolerance)
    
    # IMPLEMENT BATCH GRADIENT DESCENT AND RETURN THE OPTIMAL PARAMETERS
    def runBatchGD(self, theta, learning_rate=0.001, total_epocs=100000, loss=None, thetas=None, theta_0s=None, tolerance=1e-6):
        if loss is None: loss = []
        if thetas is None: thetas = []
        if theta_0s is None: theta_0s = []
        epocs = 0

        while(True):
            # at time step t = 0, take gradient of J wrt THETA
            error = self.get_total_error(theta)
            loss.append( error )
            theta_0s.append(theta[0][0])
            thetas.append(theta[1][0])

            grad = self.get_gradient(theta)
            theta = theta - learning_rate*grad

            if epocs > 0:
                isConvergeCriteriaTrue = self.isConvergeCriteriaMet(
                    epocs, total_epocs, loss[epocs-1], loss[epocs], tolerance
                )
            else:
                isConvergeCriteriaTrue = False


            if epocs % 200 == 0 or isConvergeCriteriaTrue == True:
                print(f"Epoch {epocs}, Loss: {loss[epocs]}")
                print(f"Theta : {theta}")

            # Convergence criteria
            if(isConvergeCriteriaTrue == True):
                error = self.get_total_error(theta)
                loss.append( error )
                theta_0s.append(theta[0][0])
                thetas.append(theta[1][0])
                break

            epocs += 1

        return theta, loss, {  
            "thetas" : thetas,
            "theta_0s" : theta_0s
        }
    # preprocessing for 3A, 3B and 4
    def preprocessing(self,params):
        print("theta range from ",min(params['thetas']), " to ",max(params['thetas']))
        print("theta_0 range from ",min(params['theta_0s']), " to ",max(params['theta_0s']))

        # Get x and y axis
        theta1_vals = np.linspace(min(params['thetas'])-30, max(params['thetas'])+30, 200)
        theta0_vals = np.linspace(min(params['theta_0s'])-30, max(params['theta_0s'])+30, 200)
        T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
        J_vals = np.zeros((T0.shape[0], T0.shape[1]))

        # compute J on grid
        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                theta = np.array([[T0[i, j]], [T1[i, j]]])
                J_vals[i, j] = self.get_total_error(theta)

        # Initial point
        theta = np.array([[theta0_vals[0]], [theta1_vals[0]]])

        # initialize prev and current cost
        prevCost = 0
        currentCost = self.get_total_error(theta)

        return theta, prevCost, currentCost, T0, T1, J_vals
    
    def getUpdatedThetaAndCurrentCost(self, thetaOld, learning_rate, currentCost):
        grad = self.get_gradient(thetaOld)
        thetaUpdated = thetaOld - learning_rate*grad
        prevCost = currentCost
        currentCost = self.get_total_error(thetaUpdated)

        return prevCost, thetaUpdated, currentCost
    
    def getUpdateCounterSize(self, learning_rate):
        s = str(learning_rate)
        if '.' in s:
            decimals = len(s.split('.')[1])
        else:
            decimals = 0

        if decimals == 3:
            return 200
        elif decimals == 2:
            return 20
        else:
            return 10
    
    def get3DGradientDescentPlot(self, params, learning_rate=0.1, iterations=2000, tolerance=1e-9, sleepSeconds = 0.02):
        # 3A and 3B
        preprocessing_results = self.preprocessing(params)
        theta, prevCost, currentCost, T0, T1, J_vals = preprocessing_results

        # Plot 3D surface
        fig = plt.figure(figsize=(10, 7))
        threeDAX = fig.add_subplot(111, projection='3d')
        surf = threeDAX.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.5)
        fig.colorbar(
            surf,
            shrink=0.5,
            aspect=5,
            pad=0.1,
            label=r'Cost=J($\theta_0$, $\theta_1$)'
        )

        threeDAX.set_xlabel(r'$\theta_0$(Bias)')
        threeDAX.set_ylabel(r'$\theta_1$')
        threeDAX.set_zlabel(r'$J(\theta_0, \theta_1)$')

        # Starting point
        point = threeDAX.scatter(theta[0][0], theta[1][0], currentCost, c="r", s=50)

        # Trail line
        path_x, path_y, path_z = [theta[0][0]], [theta[1][0]], [currentCost]
        path_line, = threeDAX.plot(path_x, path_y, path_z, 'r--')

        # Show figure once
        handle = display(fig, display_id=True)

        for i in range(iterations):
            prevCost, theta, currentCost = self.getUpdatedThetaAndCurrentCost(theta, learning_rate, currentCost)

            # Update point
            point._offsets3d = (theta[0], theta[1], [currentCost])
            # Update trail
            path_x.append(theta[0][0])
            path_y.append(theta[1][0])
            path_z.append(currentCost)
            path_line.set_data(np.array(path_x), np.array(path_y))
            path_line.set_3d_properties(np.array(path_z))
            if i % 20 == 0:  # only update plot every 20 steps
                handle.update(fig) # Update same figure (no new images)
                time.sleep(sleepSeconds)

            isConvergeCriteriaTrue = self.isConvergeCriteriaMet(
                i, iterations, prevCost, currentCost, tolerance
            )

            if i % 100 == 0 or isConvergeCriteriaTrue:
                print(f"Epoch {i}, Loss: {currentCost}")
                print(f"Theta : {theta}")

            if isConvergeCriteriaTrue:
                break

        plt.close(fig)
        
    def getContourPlot(self, params, learning_rate=0.1, iterations=2000, tolerance=1e-9, optSleepStep=False, sleepSeconds = 0.02):
        preprocessing_results = self.preprocessing(params)
        theta, prevCost, currentCost, T0, T1, J_vals = preprocessing_results

        # Plot
        fig, contourAX = plt.subplots(figsize=(16, 7))

        # Draw contour background
        contours = contourAX.contour(
            T0, T1, J_vals, levels=30, cmap='viridis', linewidths=1.2
        )
        fig.colorbar(
            contours,
            shrink=0.5,
            aspect=5,
            pad=0.1,
            label=r'Cost=J($\theta_0$, $\theta_1$)'
        )

        plt.ion()
        contourAX.set_xlabel(r'$\theta_0$')
        contourAX.set_ylabel(r'$\theta_1$')
        contourAX.set_title(
            rf'Contour Plot of $J(\theta_0, \theta_1)$ with learning rate = {learning_rate}'
        )

        # Initial point
        point = contourAX.scatter(theta[0][0], theta[1][0], c="r", s=30)

        # Trail line
        path_x, path_y, path_z = [theta[0][0]], [theta[1][0]], [currentCost]
        path_line, = contourAX.plot(path_x, path_y, 'r--')

        # Show figure once
        handle = display(fig, display_id=True)

        for i in range(iterations):
            prevCost, theta, currentCost = self.getUpdatedThetaAndCurrentCost(
                theta, learning_rate, currentCost
            )

            # Update point
            point.set_offsets(np.c_[theta[0], theta[1]])

            # Update trail
            path_x.append(theta[0][0])
            path_y.append(theta[1][0])
            path_line.set_data(path_x, path_y)

            # Plot the contour movement
            contourAX.scatter(theta[0][0], theta[1][0], c="r", s=30)
            
            updateCounterSize = self.getUpdateCounterSize(learning_rate)

            if i % updateCounterSize == 0:  # only update plot every 20 steps
                handle.update(fig) # Update same figure (no new images)
                time.sleep(sleepSeconds)

            isConvergeCriteriaTrue = self.isConvergeCriteriaMet(
                i, iterations, prevCost, currentCost, tolerance
            )

            if i % 100 == 0 or isConvergeCriteriaTrue:
                print(f"Epoch {i}, Loss: {currentCost}")
                print(f"Theta : {theta}")

            if isConvergeCriteriaTrue:
                break

        plt.ioff()
        plt.close(fig)
