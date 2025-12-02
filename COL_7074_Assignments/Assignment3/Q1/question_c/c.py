#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from prunningUtils import post_prune_bottom_up


def main(train_path, val_path, test_path, output_path):

    print("Loading CSV files")
    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    print(f"Train shape      : {train_data.shape}")
    print(f"Validation shape : {validation_data.shape}")
    print(f"Test shape       : {test_data.shape}")
    print("--------------------------------------")

    # Step 1: One-Hot Encoding
    print("\nPerforming One-Hot Encoding...")
    train_data_enc, val_data_enc, test_data_enc = convertCategoricalColumnsToOneHotEncoding(
        train_data, validation_data, test_data
    )

    # Step 2: Preprocess Inputs
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    print("\nPreprocessing numeric features...")
    trainX, trainY, valX, valY, testX, testY = pre_process(
        train_data_enc,
        val_data_enc,
        test_data_enc,
        categorical_cols=[],
        continuous_cols=continuous_cols
    )

    # Collect newly encoded categorical columns
    new_categorical_cols = list(trainX.columns[trainX.dtypes == bool])
    new_categorical_cols.extend(["toss", "bat_first", "format"])

    print(f"Total categorical cols: {len(new_categorical_cols)}")
    print("--------------------------------------")

    categorical_cols = new_categorical_cols
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    # Step 3: Decision Tree + Post-Pruning
    maxDepths = [15, 25, 35, 45]
    expectedPrunedDepth = 35

    for depth in maxDepths:
        print(f"\nBuilding Decision Tree (maxDepth={depth})")
        model = DecisionTreeModel(
            depth,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols
        )

        print("→ Training full tree...")
        model.fit(trainX, trainY)

        print("→ Starting Greedy Post-Pruning...")
        history = post_prune_bottom_up(
            model,
            trainX, trainY,
            valX, valY,
            testX, testY
        )

        # unpack pruning history
        nodes, train_accs, val_accs, test_accs = zip(*history)

        print("→ Plotting pruning curve")
        plotA(
            nodes,
            train_accs,
            val_accs,
            test_accs,
            f"plots/postPruneCurve{depth}.png",
            xLabel="Number of Nodes in Tree",
            title=f"Post-Pruning Curve (maxDepth={depth})"
        )
        if( depth == expectedPrunedDepth):
            best_predictY_test = model.predict(testX)
            save_predictions_to_csv(best_predictY_test, output_path)

    print("\nDone.")



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage:")
        print(" python3 c.py <train.csv> <validation.csv> <test.csv> <output_dir>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    main(train_path, val_path, test_path, output_path)
