import numpy as np
from modelUtils import getProbsParameters

class NaiveBayesCustom:
    def __init__(self):
        self.vocabularyTitle = None
        self.vocabularyContent = None
        self.params = None

    def setVocabulary(self, vocabularyTitle, vocabularyContent):
        """Set separate vocabularies for title and content."""
        self.vocabularyTitle = vocabularyTitle
        self.vocabularyContent = vocabularyContent

    def setParams(self, params):
        self.params = params

    def fit(
        self, 
        df, 
        smoothening, 
        class_col="Class Index", 
        text_col1="Tokenized Title", 
        text_col2="Tokenized Description",
        new_feature_cols=[]
    ):
        """
        Learn separate parameters θ(title) and θ(content) for each class.
        """
        classes = set(df[class_col])
        num_classes = len(classes)
        num_examples = len(df)
        num_words_title = len(self.vocabularyTitle)
        num_words_content = len(self.vocabularyContent)

        print(f"Training with {num_classes} classes, {num_examples} examples.")
        print(f"Title vocab size: {num_words_title}, Content vocab size: {num_words_content}")

        # Compute priors and conditionals separately for title and content
        phi_y, phi_j_given_y_title = getProbsParameters(
            df, smoothening, num_words_title, num_classes, num_examples, 
            class_col, text_col1, self.vocabularyTitle
        )

        _, phi_j_given_y_content = getProbsParameters(
            df, smoothening, num_words_content, num_classes, num_examples, 
            class_col, text_col2, self.vocabularyContent
        )

        # Compute Gaussian parameters for each numeric new_feature
        new_feature_stats = {col: [] for col in new_feature_cols}
        for c in classes:
            subset = df[df[class_col] == c]
            for col in new_feature_cols:
                vals = subset[col].values
                mean = np.mean(vals)
                std = np.std(vals) + 1e-6
                new_feature_stats[col].append((mean, std))

        self.params = {
            "phi_y": phi_y,
            "phi_j_given_y_title": phi_j_given_y_title,
            "phi_j_given_y_content": phi_j_given_y_content,
            "new_feature_stats": new_feature_stats,
            "new_feature_cols": new_feature_cols
        }

    def predict(
        self, 
        df, 
        text_col1="Tokenized Title", 
        text_col2="Tokenized Description", 
        predicted_col="Predicted"
    ):
        """
        Predict the most likely class by combining log-probabilities
        from both title and content (independent assumption).
        """
        phi_y = self.params["phi_y"]
        phi_j_given_y_title = self.params["phi_j_given_y_title"]
        phi_j_given_y_content = self.params["phi_j_given_y_content"]
        new_feature_stats = self.params["new_feature_stats"]
        new_feature_cols = self.params["new_feature_cols"]

        num_classes = len(phi_y)
        predictions = []

        for _, row in df.iterrows():
            tokensOfTitle = row[text_col1]
            tokensOfContent = row[text_col2]
            log_probs = np.zeros(num_classes)

            for class_index in range(num_classes):
                log_prob = np.log(phi_y[class_index])

                # Title contribution
                for token in tokensOfTitle:
                    if token in self.vocabularyTitle:
                        idx = self.vocabularyTitle[token]
                        log_prob += np.log(phi_j_given_y_title[class_index][idx])
                    else:
                        log_prob += np.log(1e-10)  # unseen word smoothing

                # Content contribution
                for token in tokensOfContent:
                    if token in self.vocabularyContent:
                        idx = self.vocabularyContent[token]
                        log_prob += np.log(phi_j_given_y_content[class_index][idx])
                    else:
                        log_prob += np.log(1e-10)

                # Numeric new_feature likelihoods (Gaussian)
                for col in new_feature_stats:
                    val = row[col]
                    mean, std = new_feature_stats[col][class_index]
                    gaussian_loglik = (
                        -0.5 * np.log(2 * np.pi * std**2)
                        - ((val - mean) ** 2) / (2 * std**2)
                    )
                    log_prob += gaussian_loglik

                log_probs[class_index] = log_prob

            predictions.append(np.argmax(log_probs))

        df[predicted_col] = predictions
        print(f"Predictions added to column '{predicted_col}'.")
        return df