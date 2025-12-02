#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from utils import (
    convertCategoricalColumnsToOneHotEncoding,
    pre_process,
    evaluate,
    plotA,
    save_predictions_to_csv
)
from decisionTree import DecisionTreeModel


def main(train_path, val_path, test_path, output_path):

    print("Loading CSV files")
    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    print(f"Train shape      : {train_data.shape}")
    print(f"Validation shape : {validation_data.shape}")
    print(f"Test shape       : {test_data.shape}")

    # Step 1: One-Hot Encoding
    print("Performing One-Hot Encoding...")
    train_data_encoded, val_data_encoded, test_data_encoded = (
        convertCategoricalColumnsToOneHotEncoding(train_data, validation_data, test_data)
    )

    # Step 2: Preprocess (trainX/trainY/...)
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    print("Preprocessing continuous features...")
    trainX, trainY, valX, valY, testX, testY = pre_process(
        train_data_encoded,
        val_data_encoded,
        test_data_encoded,
        categorical_cols=[],
        continuous_cols=continuous_cols
    )

    # categorical cols found after encoding
    new_categorical_cols = list(trainX.columns[trainX.dtypes == bool])
    new_categorical_cols.extend(["toss", "bat_first", "format"])

    print(f"Total categorical cols after processing: {len(new_categorical_cols)}")

    # Step 3: Decision Tree Experiments
    maxDepths = [15, 25, 35, 45]
    trainAccuracies = []
    valAccuracies = []
    testAccuracies = []

    categorical_cols = new_categorical_cols
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    best_model = None
    best_validation_accuracy = 0
    best_max_depth = 0

    for depth in maxDepths:
        print(f"\n Testing Decision Tree with maxDepth = {depth}")
        model = DecisionTreeModel(
            depth,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols
        )

        print("→ Training model")
        model.fit(trainX, trainY)

        print("→ Predicting Train Data")
        pred_train = model.predict(trainX)
        trainAccuracies.append(evaluate(pred_train, trainY))

        print("→ Predicting Validation Data")
        pred_val = model.predict(valX)
        val_accuracy = evaluate(pred_val, valY)
        valAccuracies.append(val_accuracy)

        print("→ Predicting Test Data")
        pred_test = model.predict(testX)
        testAccuracies.append(evaluate(pred_test, testY))

        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_max_depth = depth
            best_model = model

    # Step 4: Plot the results
    print("\nGenerating plot...")
    plotA(maxDepths, trainAccuracies, valAccuracies, testAccuracies, "plots/decisionTreeDepthVsAccuracy.png")

    print(f"Best validation accuracy is coming to be {best_validation_accuracy} and for the depth {best_max_depth}")
    best_predictY_test = best_model.predict(testX)
    save_predictions_to_csv(best_predictY_test, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage:")
        print(" python3 a.py <train.csv> <validation.csv> <test.csv> <output_dir>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    main(train_path, val_path, test_path, output_path)
