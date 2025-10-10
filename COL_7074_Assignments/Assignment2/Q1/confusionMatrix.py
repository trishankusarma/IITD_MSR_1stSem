import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = np.unique(np.concatenate((self.y_true, self.y_pred)))
        self.matrix = self._compute_matrix()

    def _compute_matrix(self):
        n = len(self.labels)
        mat = np.zeros((n, n), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        for t, p in zip(self.y_true, self.y_pred):
            i = label_to_index[t]
            j = label_to_index[p]
            mat[i, j] += 1

        return pd.DataFrame(mat, index=self.labels, columns=self.labels)

    def plot(self):

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def most_correct_category(self):
        diagonal_values = np.diag(self.matrix)
        best_class_index = np.argmax(diagonal_values)
        return self.labels[best_class_index], diagonal_values[best_class_index]

    def report(self):
        total = np.sum(self.matrix.values)
        correct = np.trace(self.matrix.values)
        accuracy = correct / total * 100

        best_class, best_value = self.most_correct_category()

        print("=== Confusion Matrix Report ===")
        print(self.matrix)
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print(f"Category with highest diagonal value: {best_class} ({best_value} correct samples)")
        print(f"Interpretation: Model predicts '{best_class}' most correctly among all classes.")
