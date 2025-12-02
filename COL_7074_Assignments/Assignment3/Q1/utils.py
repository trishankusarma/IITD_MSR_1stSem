import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

# ---------------------- Preprocessing / One-hot / Evaluation ----------------------
def convertCategoricalColumnsToOneHotEncoding(train_data, validation_data, test_data, add_day_match_if_missing=False):
    # Work on copies to avoid mutating original dfs
    train = train_data.copy()
    val = validation_data.copy()
    test = test_data.copy()

    # Identify categorical cols (object dtype) + optionally include day_match if present as int
    categorical_cols = list(train.select_dtypes(include=['object']).columns)
    if add_day_match_if_missing and "day_match" in train.columns and "day_match" not in categorical_cols:
        categorical_cols.append("day_match")

    # One-hot encode datasets
    train_encoded = pd.get_dummies(train, columns=categorical_cols)
    val_encoded = pd.get_dummies(val, columns=categorical_cols)
    test_encoded = pd.get_dummies(test, columns=categorical_cols)

    # Align all columns together (outer join) so every dataset has same columns
    # Use outer join and fill_value 0 ensures any missing column becomes zero
    train_encoded, val_encoded = train_encoded.align(val_encoded, join='outer', axis=1, fill_value=0)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join='outer', axis=1, fill_value=0)
    val_encoded, test_encoded = val_encoded.align(test_encoded, join='outer', axis=1, fill_value=0)

    return train_encoded, val_encoded, test_encoded

def pre_process(train, val, test, categorical_cols, continuous_cols):
    trainX = train.drop(columns=["result"])
    trainY = train["result"]

    valX = val.drop(columns=["result"])
    valY = val["result"]

    testX = test.drop(columns=["result"])
    testY = test["result"]

    # normalize continuous
    for col in continuous_cols:
        m = trainX[col].mean()
        s = trainX[col].std()

        trainX[col] = (trainX[col] - m) / s
        valX[col] = (valX[col] - m) / s
        testX[col] = (testX[col] - m) / s

    return trainX, trainY, valX, valY, testX, testY

def normalize_continuous(trainX, valX, testX, continuous_cols):
    train = trainX.copy()
    val = valX.copy()
    test = testX.copy()
    for col in continuous_cols:
        if col not in train.columns:
            continue
        mean = train[col].mean()
        std = train[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # avoid division by zero: set to zero column
            train[col] = 0.0
            val[col] = 0.0
            test[col] = 0.0
            continue
        train[col] = (train[col] - mean) / std
        val[col] = (val[col] - mean) / std
        test[col] = (test[col] - mean) / std
    return train, val, test

def evaluate(predictY, actualY, verbose=True):
    predictY = np.array(predictY, dtype=int)
    actualY = np.array(actualY, dtype=int)
    if predictY.shape != actualY.shape:
        raise ValueError("predict and actual shapes mismatch")

    TP = int(np.sum((predictY == 1) & (actualY == 1)))
    TN = int(np.sum((predictY == 0) & (actualY == 0)))
    FP = int(np.sum((predictY == 1) & (actualY == 0)))
    FN = int(np.sum((predictY == 0) & (actualY == 1)))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1Score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if verbose:
        print(f" Accuracy : {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1Score:.4f}")
    return accuracy

def plotA(xAxis, train_accuracies, validation_accuracies, test_accuracies, file_path,
          xLabel='Max Depth of Decision Tree', title='Decision Tree Depth vs Accuracy'):
    plt.figure(figsize=(8,5))
    plt.plot(xAxis, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(xAxis, validation_accuracies, marker='o', label='Validation Accuracy')
    plt.plot(xAxis, test_accuracies, marker='o', label='Test Accuracy')
    plt.xlabel(xLabel)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
    print(f"Saved plot to {file_path}")

def save_predictions_to_csv(pred_list, csv_path):
    df = pd.DataFrame({"result": pred_list})
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")