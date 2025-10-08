import numpy as np

class NaiveBayes:
    def __init__(self):
        pass
    
    def setVocabulary(self, vocabulary):
        self.vocabulary = vocabulary
    
    def setParams(self, params):
        self.params = params
    
    def getProbsParameters(self, df, smoothening, num_words, num_classes, num_examples, class_col, text_col):
        freq_y = np.zeros(num_classes) + smoothening
        phi_j_given_y = np.zeros((num_classes, num_words)) + smoothening
        
        for _, row in df.iterrows():
            class_label = row[class_col]  # classes are 1-indexed
            freq_y[class_label] += 1
            for word_j in row[text_col]:
                if word_j in self.vocabulary:
                    word_j_index = self.vocabulary[word_j]
                    phi_j_given_y[class_label][word_j_index] += 1

        # Normalize conditionals
        for class_index in range(num_classes):
            phi_j_given_y[class_index] /= np.sum(phi_j_given_y[class_index])

        # Normalize priors
        phi_y = freq_y / np.sum(freq_y)
        
        print(f"Shape of phi_y: {phi_y.shape}")
        print(f"Shape of phi_j_given_y: {phi_j_given_y.shape}")
        return phi_y, phi_j_given_y
        
    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        classes = set(df[class_col])
        num_classes = len(classes)
        num_examples = len(df)
        num_words = len(self.vocabulary)
        
        print(f"Number of classes: {num_classes}, examples: {num_examples}, vocab size: {num_words}")
        
        phi_y, phi_j_given_y = self.getProbsParameters(df, smoothening, num_words, num_classes, num_examples, class_col, text_col)
        
        self.params = {"phi_y": phi_y, "phi_j_given_y": phi_j_given_y}
        pass
    
    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        phi_y, phi_j_given_y = self.params["phi_y"], self.params["phi_j_given_y"]
        num_classes = len(phi_y)
        predictions = []
        
        for _, row in df.iterrows():
            tokens = row[text_col]
            log_probs = np.zeros(num_classes)
            
            for class_index in range(num_classes):
                log_prob = np.log(phi_y[class_index])
                for token in tokens:
                    if token in self.vocabulary:
                        word_idx = self.vocabulary[token]
                        log_prob += np.log(phi_j_given_y[class_index][word_idx])
                    else:
                        log_prob += np.log(1e-10)  # unseen word smoothing
                log_probs[class_index] = log_prob
            
            predictions.append(np.argmax(log_probs))
        
        df[predicted_col] = predictions
        pass
    
    def evaluate(self, predicted_col, actual_col, num_test_examples):
        """
        Evaluate the model predictions.

        Args:
            predicted_col (list): List of predicted class labels.
            actual_col (list): List of actual class labels.
            num_test_examples (int): Number of test examples.

        Returns:
            metrics (dict): Dictionary containing overall accuracy, 
                            per-class precision, recall, F1, and macro F1.
        """
        print(f"Evaluating on {num_test_examples} examples")
        # Get list of unique classes
        classes = list(set(actual_col))

        # Initialize counters for each class
        true_positive = {cls: 0 for cls in classes}
        false_positive = {cls: 0 for cls in classes}
        false_negative = {cls: 0 for cls in classes}

        correct = 0

        # Compute counts
        for i in range(num_test_examples):
            pred = predicted_col[i]
            actual = actual_col[i]
            if pred == actual:
                correct += 1
                true_positive[actual] += 1
            else:
                false_positive[pred] += 1
                false_negative[actual] += 1

        # Overall accuracy
        accuracy = correct / num_test_examples * 100
        print("Overall Accuracy: {:.2f}%\n".format(accuracy))

        # Compute precision, recall, F1 per class and macro F1
        metrics_per_class = {}
        f1_list = []

        for cls in classes:
            tp = true_positive[cls]
            fp = false_positive[cls]
            fn = false_negative[cls]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            f1_list.append(f1)
            metrics_per_class[cls] = {'precision': precision, 'recall': recall, 'f1': f1}

            print("Class {} -> Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(
                cls, precision, recall, f1))

        # Macro-average F1
        macro_f1 = sum(f1_list) / len(f1_list)
        print("\nMacro-Average F1 Score: {:.4f}".format(macro_f1))

        # Return metrics as dictionary
        metrics = {
            'overall_accuracy': accuracy,
            'per_class': metrics_per_class,
            'macro_f1': macro_f1
        }

        return metrics
