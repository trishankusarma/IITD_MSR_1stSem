import numpy as np

# For single text field (e.g., Description only)
def run_model(model, vocabulary, trainingData, testingData, smoothening=1.0, text_col="Tokenized Description"):
    # Build the Naive Bayes Model
    model.setVocabulary(vocabulary)

    # Training
    model.fit(trainingData, smoothening, text_col=text_col)  # Laplace smoothening parameter used: 1.0

    # Predict on training data
    model.predict(trainingData, text_col=text_col)
    print("Evaluating on train data...")
    evaluate(trainingData["Predicted"], trainingData["Class Index"], trainingData.shape[0])

    # Predict on testing data
    model.predict(testingData, text_col=text_col)
    print("Evaluating on test data...")
    evaluate(testingData["Predicted"], testingData["Class Index"], testingData.shape[0])
    pass


# For considering both Title and Content separately
def run_model2(model, vocabularyTitle, vocabularyContent, trainingData, testingData,
               smoothening=1.0, text_col1="Tokenized Title", text_col2="Tokenized Description"):
    # Build the Naive Bayes Model
    model.setVocabulary(vocabularyTitle, vocabularyContent)

    # Training
    model.fit(trainingData, smoothening, text_col1=text_col1, text_col2=text_col2)

    # Predict on training data
    model.predict(trainingData, text_col1=text_col1, text_col2=text_col2)
    print("Evaluating on train data...")
    evaluate(trainingData["Predicted"], trainingData["Class Index"], trainingData.shape[0])

    # Predict on testing data
    model.predict(testingData, text_col1=text_col1, text_col2=text_col2)
    print("Evaluating on test data...")
    evaluate(testingData["Predicted"], testingData["Class Index"], testingData.shape[0])
    pass


def getProbsParameters(df, smoothening, num_words, num_classes, num_examples, class_col, text_col, vocabulary):
    """
    Compute Naive Bayes parameters: class priors and conditional probabilities P(word|class).
    """
    freq_y = np.zeros(num_classes) + smoothening
    phi_j_given_y = np.zeros((num_classes, num_words)) + smoothening

    for _, row in df.iterrows():
        class_label = row[class_col]  # assuming classes are 0-indexed
        freq_y[class_label] += 1
        for word_j in row[text_col]:
            if word_j in vocabulary:
                word_j_index = vocabulary[word_j]
                phi_j_given_y[class_label][word_j_index] += 1

    # Normalize conditionals
    phi_j_given_y /= np.sum(phi_j_given_y, axis=1, keepdims=True)

    # Normalize priors
    phi_y = freq_y / np.sum(freq_y)

    print(f"Shape of phi_y: {phi_y.shape}")
    print(f"Shape of phi_j_given_y: {phi_j_given_y.shape}")
    return phi_y, phi_j_given_y


def evaluate(predicted_col, actual_col, num_test_examples):
    """
    Evaluate model predictions using accuracy, precision, recall, F1, and macro F1.
    """
    print(f"Evaluating on {num_test_examples} examples")

    classes = list(set(actual_col))

    # Initialize counters
    true_positive = {cls: 0 for cls in classes}
    false_positive = {cls: 0 for cls in classes}
    false_negative = {cls: 0 for cls in classes}

    correct = 0

    # Count TP, FP, FN
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

    # Per-class metrics
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