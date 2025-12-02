import numpy as np
import logging

np.seterr(over='ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Activation Functions
# ---------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]

# ---------------------------
# Loss
# ---------------------------

def cross_entropy_loss(y, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y * np.log(y_pred), axis=1))


# ---------------------------
# Neural Network
# ---------------------------

class NeuralNetwork:
    def __init__(self, hidden_layers=None, output_size=36, learning_rate=0.01, input_size=3072):

        if hidden_layers is None:
            hidden_layers = []

        self.learning_rate = learning_rate

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.L = len(layer_sizes) - 1

        self.params = {}
        for i in range(self.L):
            self.params[f"W{i}"] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1 / layer_sizes[i])
            self.params[f"b{i}"] = np.zeros((1, layer_sizes[i+1]), dtype=np.float32)

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, X, use_relu=False):

        cache = {"A0": X}

        for i in range(self.L - 1):
            Z = cache[f"A{i}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"Z{i+1}"] = Z

            if use_relu:
                cache[f"A{i+1}"] = relu(Z)
            else:
                cache[f"A{i+1}"] = sigmoid(Z)

        # Output layer
        ZL = cache[f"A{self.L-1}"] @ self.params[f"W{self.L-1}"] + self.params[f"b{self.L-1}"]
        cache[f"Z{self.L}"] = ZL
        cache[f"A{self.L}"] = softmax(ZL)

        return cache[f"A{self.L}"], cache

    # ---------------------------
    # Backward
    # ---------------------------
    def backward(self, cache, y_true, use_relu=False):

        grads = {}
        m = y_true.shape[0]

        # Output layer derivative
        dZ = cache[f"A{self.L}"] - y_true
        grads[f"dW{self.L-1}"] = (cache[f"A{self.L-1}"].T @ dZ) / m
        grads[f"db{self.L-1}"] = np.sum(dZ, axis=0, keepdims=True) / m

        # Hidden layers
        for i in reversed(range(self.L - 1)):
            dA = dZ @ self.params[f"W{i+1}"].T

            if use_relu:
                dZ = dA * relu_derivative(cache[f"Z{i+1}"])
            else:
                dZ = dA * sigmoid_derivative(cache[f"A{i+1}"])

            grads[f"dW{i}"] = (cache[f"A{i}"].T @ dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    # ---------------------------
    # Update
    # ---------------------------
    def update(self, grads):
        for k in grads:
            w = k.replace("d", "")
            self.params[w] -= self.learning_rate * grads[k]

    # ---------------------------
    # Fit (with only plateau stopping)
    # ---------------------------
    def fit(self, X, y, epochs=400, batch_size=32, val_split=0.1,
        loss_delta_threshold=1e-4, use_relu=False):

        logging.info(f"ReLU: {use_relu}")
    
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
    
        # Split validation
        val_size = int(val_split * len(X))
        X_val, y_val = X[:val_size], y[:val_size]
        X_train, y_train = X[val_size:], y[val_size:]
    
        num_classes = self.params[f"b{self.L - 1}"].shape[1]
    
        y_train_oh = one_hot_encode(y_train, num_classes)
        y_val_oh = one_hot_encode(y_val, num_classes)
    
        prev_train_loss = None
        prev_val_loss = None
    
        for epoch in range(epochs):
    
            # Shuffle each epoch
            perm = np.random.permutation(len(X_train))
            X_train, y_train, y_train_oh = X_train[perm], y_train[perm], y_train_oh[perm]
    
            batch_losses = []
    
            # Mini-batch SGD
            for i in range(0, len(X_train), batch_size):
                Xb = X_train[i:i + batch_size]
                yb = y_train_oh[i:i + batch_size]
    
                preds, cache = self.forward(Xb, use_relu)
                loss = cross_entropy_loss(yb, preds)
                batch_losses.append(loss)
    
                grads = self.backward(cache, yb, use_relu)
                self.update(grads)
    
            # Epoch metrics
            train_loss = np.mean(batch_losses)
            val_preds, _ = self.forward(X_val, use_relu)
            val_loss = cross_entropy_loss(y_val_oh, val_preds)
    
            if epoch % 50 == 0:
                train_acc = np.mean(np.argmax(self.forward(X_train, use_relu)[0], axis=1) == y_train) * 100
                val_acc = np.mean(np.argmax(val_preds, axis=1) == y_val) * 100
                logging.info(f"Epoch {epoch} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")
    
            # ---- Plateau stopping ----
            if prev_train_loss is not None:
                if (abs(train_loss - prev_train_loss) < loss_delta_threshold and
                    abs(val_loss - prev_val_loss) < loss_delta_threshold):
    
                    logging.info(f"Early Stop: plateau at epoch {epoch}")
                    break
    
            prev_train_loss = train_loss
            prev_val_loss = val_loss

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, X, use_relu=False):
        probs, _ = self.forward(X, use_relu=use_relu)
        return np.argmax(probs, axis=1)
