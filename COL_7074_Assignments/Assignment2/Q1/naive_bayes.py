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
