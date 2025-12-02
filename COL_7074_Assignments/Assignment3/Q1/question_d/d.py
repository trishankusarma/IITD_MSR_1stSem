import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
    # ------------------------- Load CSVs ------------------------------
    print("Loading data files ...")

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    print(f"Shape of train_data : {train_data.shape}")
    print(f"Shape of validation_data : {validation_data.shape}")
    print(f"Shape of test_data : {test_data.shape}")

    # --------------------- One-Hot Encoding ----------------------------
    print("Encoding categorical columns ...")

    train_encoded, val_encoded, test_encoded = convertCategoricalColumnsToOneHotEncoding(
        train_data, validation_data, test_data
    )

    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    trainX, trainY, valX, valY, testX, testY = pre_process(
        train_encoded,
        val_encoded,
        test_encoded,
        categorical_cols=[],
        continuous_cols=continuous_cols,
    )

    # --------------------- Category + Continuous Info -------------------
    new_categorical_cols = list(trainX.columns[trainX.dtypes == bool])
    new_categorical_cols.extend(["toss", "bat_first", "format"])

    categorical_cols = new_categorical_cols
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    maxDepths = [15, 25, 35, 45]
    expectedPrunedDepth = 35

    # --------------------- Model Training Loop --------------------------
    for depth in maxDepths:
        print(f"Testing maxDepth = {depth} using Gini index")

        model = DecisionTreeModel(
            depth,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            using_gini_index=True,
        )

        print("Training Decision Tree ...")
        model.fit(trainX, trainY)

        print("Post-Pruning the tree (Greedy) ...")
        history = post_prune_bottom_up(model, trainX, trainY, valX, valY, testX, testY)

        nodes, train_accs, val_accs, test_accs = zip(*history)

        plot_title = f"Post-Pruning Curve (maxDepth={depth})"

        # Save plot to output directory
        plot_file = f"plots/prune_curve_depth_{depth}.png"
        plt.figure()
        plotA(nodes, train_accs, val_accs, test_accs, plot_file,
              xLabel="Number of Nodes in Tree",
              title=plot_title)

        print(f"Saved plot: {plot_file}")

        if( depth == expectedPrunedDepth):
            best_predictY_test = model.predict(testX)
            save_predictions_to_csv(best_predictY_test, output_path)

    print("\nFinished all depths!")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 question_d.py <train.csv> <validation.csv> <test.csv> <output_dir>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    main(train_path, val_path, test_path, output_path)
