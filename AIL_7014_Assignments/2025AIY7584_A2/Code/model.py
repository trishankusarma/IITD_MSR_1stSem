import numpy as np
import matplotlib.pyplot as plt
import time

class ModelClass:
    # ------------------- Data Augmentation -------------------
    def augmentData(self, X_train, X_test):
        num_examples_train, num_examples_test = X_train.shape[0], X_test.shape[0]
        
        # augmenting trainX and testX with a new column of 1
        X_train_AUG = np.hstack((np.ones((num_examples_train, 1)), X_train))
        X_test_AUG = np.hstack((np.ones((num_examples_test, 1)), X_test))
        
        print("Shape of trainX after augmenting a new column of 1:", X_train_AUG.shape)
        print("Shape of testX after augmenting a new column of 1:", X_test_AUG.shape)
        
        return X_train_AUG, X_test_AUG
    
    # ------------------- Parameter Initialization -------------------
    def initializeParameters(self, num_features):
        W = np.ones(num_features)
        print("Shape of W:", W.shape)
        return W
    
    # ------------------- Lasso Objective -------------------
    def lasso_objective_loss(self, X, y, w, num_examples, alpha=0.2):
        residual = X @ w - y
        loss = (0.5 / num_examples) * np.dot(residual, residual)
        reg = alpha * np.linalg.norm(w, 1)
        return loss + reg
    
    # ------------------- Gradients -------------------
    def gradOfSmoothPart(self, X, y, w, num_examples):
        if X.ndim == 1:
            X = X.reshape(1, -1)
            y = np.array([y])
        return (1 / num_examples) * X.T @ (X @ w - y)
    
    def gradOfRegularizationPart(self, w, alpha=0.2):
        grad = alpha * np.sign(w)
        grad[0] = 0   # don't regularize bias
        return grad
    
    def grad(self, X, y, w, num_examples):
        return self.gradOfSmoothPart(X, y, w, num_examples) + self.gradOfRegularizationPart(w)
    
    # ------------------- Proximal Operator -------------------
    def getProx(self, v, step_size, alpha):
        out = np.sign(v) * np.maximum(np.abs(v) - step_size * alpha, 0.0)
        # do not shrink bias term
        v[0] = out[0]
        return out
    
    # ------------------- Convergence Check -------------------
    def check_convergence(self, loss_history, w, prev_w, tol_loss=1e-4, tol_w=1e-4, window=10):
        """Check convergence based on rolling avg loss + parameter change."""
        converged = False
        reason = None
        
        # Condition 1: Loss stabilization
        if len(loss_history) > window:  
            avg_prev_loss = np.mean(loss_history[-(window+1):-1])  # last 10 (excluding current)
            curr_loss = loss_history[-1]                           # current loss
            if abs(avg_prev_loss - curr_loss) < tol_loss:
                converged = True
                reason = f"loss stabilized (Δ < {tol_loss})"

        # Condition 2: Parameter stability
        if np.linalg.norm(w - prev_w, 2) < tol_w:
            converged = True
            reason = f"param change < {tol_w}"
        
        return converged, reason
    
    # ------------------- Training (SGD / ProxGD / SAGA) -------------------
    def runSGD(self, X, y, w, valX, valY,
               SAGA=False, proxGD=False, backTrackLS=False,
               alpha=0.2, step_size=0.01, maxEpocs=20000,
               tol_loss=1e-4, tol_w=1e-4, window=10,
               gamma=0.02, tau=0.5,initialLearningRate=0.01, enableDynamicStepSize = True):
        
        num_examples, num_features = X.shape
        epocs = 0
        loss_history, val_loss_history, time_history = [], [], []
        start_time = time.time()
        prev_w = w.copy()
        
        print("num_examples in training set : ", num_examples)
        print("num examples in validation set : ", valX.shape[0])
        
        if SAGA:
            # Initialize stored gradients (can be zeros)
            grad_of_examples = np.zeros((num_examples, num_features))
        
        while True:
            # Shuffle data
            indices = np.random.permutation(num_examples)
            
            if enableDynamicStepSize and epocs > 10000:
                step_size = 1/np.sqrt(epocs)
            
            for i in indices:
                xi, yi = X[i, :].reshape(1, -1), y[i]
                
                if proxGD:
                    grad_w_smooth = self.gradOfSmoothPart(xi, yi, w, 1)
                    w = self.getProx(w - step_size * grad_w_smooth, step_size, alpha)
                
                else:
                    grad_w_new = self.grad(xi, yi, w, 1)
                    
                    if SAGA:
                        grad_mean = np.mean(grad_of_examples, axis=0)
                        grad_w = grad_w_new - grad_of_examples[i] + grad_mean
                        grad_of_examples[i] = grad_w_new
                    else:
                        grad_w = grad_w_new
                    
                    if backTrackLS:
                        curr_step_size = initialLearningRate
                        f_w = self.lasso_objective_loss(xi, yi, w, 1)
                        while True:
                            w_new = w - curr_step_size * grad_w
                            f_w_new = self.lasso_objective_loss(xi, yi, w_new, 1)
                            if f_w_new > f_w - gamma * curr_step_size * (grad_w.T @ grad_w):
                                curr_step_size *= tau
                            else:
                                break
                        w = w - curr_step_size * grad_w
                    else:
                        w = w - step_size * grad_w
            
            # ---- End of epoch ----
            epocs += 1
            loss = self.lasso_objective_loss(X, y, w, num_examples, alpha)
            val_loss = self.lasso_objective_loss(valX, valY, w, valX.shape[0], alpha)
            
            loss_history.append(loss)
            val_loss_history.append(val_loss)
            time_history.append(time.time() - start_time)
            
            if epocs == 1 or epocs % 500 == 0:
                print(f"Loss after {epocs} epochs: {loss:.6f}")
            
            converged, reason = self.check_convergence(loss_history, w, prev_w, tol_loss, tol_w, window=window)
            if converged:
                print(f"Converged at epoch {epocs} ({reason})")
                break
            
            if epocs >= maxEpocs:
                print(f"MaxEpocs reached ({maxEpocs}), final loss: {loss:.6f}")
                break
            
            prev_w = w.copy()
        
        # ---- Plots ----
        self.plot_loss_curve(loss_history, label="Training Loss", xlabel="Epochs", ylabel="Train Loss", title="Train Loss vs Epochs")
        self.plot_loss_curve(val_loss_history, label="Validation Loss", xlabel="Epochs", ylabel="Validation Loss", title="Validation Loss vs Epochs")
        self.plot_loss_curve(val_loss_history, x_values=time_history, label="Validation Loss", xlabel="Time Elapsed", ylabel="Validation Loss", title="Validation Loss vs Time")
        
        print("Time taken to converge:", time.time() - start_time)
        return w
    
    # ------------------- Plot -------------------
    def plot_loss_curve(self, loss_history, x_values=None, label="", xlabel="", ylabel="", title=""):
        plt.figure(figsize=(8,5))
        if x_values is None:
            plt.plot(loss_history, label=label, color="blue")
        else:
            plt.plot(x_values, loss_history, label=label, color="blue")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
