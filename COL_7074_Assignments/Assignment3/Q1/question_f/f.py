#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from utils import pre_process, save_predictions_to_csv, evaluate

def main(train_path, val_path, test_path, output_path):
    # -------------------------
    # Load data
    # -------------------------
    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    # Specify columns
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]
    categorical_cols = ["team", "opp", "host", "month"]

    # Preprocess
    trainX, trainY, valX, valY, testX, testY = pre_process(
        train_data, validation_data, test_data,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols
    )

    # -------------------------
    # Random Forest Grid Search
    # -------------------------
    n_estimators_list = [50, 150, 250, 350]
    max_features_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_samples_split_list = [2, 4, 6, 8, 10]

    best_val_acc = 0
    best_test_acc = 0
    best_params_val = {}
    best_params_test = {}
    best_model = None
    results = []

    # Create plots folder
    plots_dir = "plots"

    print("=== Starting Random Forest Grid Search ===\n")
    for n_estimators, max_features, min_samples_split in product(n_estimators_list, max_features_list, min_samples_split_list):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            criterion='entropy',
            oob_score=True,
            random_state=0,
            n_jobs=-1
        )

        clf.fit(trainX, trainY)

        train_score = accuracy_score(trainY, clf.predict(trainX))
        val_score = accuracy_score(valY, clf.predict(valX))
        test_score = accuracy_score(testY, clf.predict(testX))
        oob_score = clf.oob_score_

        results.append({
            'n_estimators': n_estimators,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'train_acc': train_score,
            'val_acc': val_score,
            'test_acc': test_score,
            'oob_acc': oob_score
        })

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} - n_estimators={n_estimators}, max_features={max_features}, min_samples_split={min_samples_split} | "
              f"Train={train_score:.4f}, Val={val_score:.4f}, Test={test_score:.4f}, OOB={oob_score:.4f}")

        # Track best validation model
        if val_score > best_val_acc:
            best_val_acc = val_score
            best_params_val = {
                'n_estimators': n_estimators,
                'max_features': max_features,
                'min_samples_split': min_samples_split
            }
            best_model = clf

        # Track best test model
        if test_score > best_test_acc:
            best_test_acc = test_score
            best_params_test = {
                'n_estimators': n_estimators,
                'max_features': max_features,
                'min_samples_split': min_samples_split
            }

    # -------------------------
    # Save best parameters
    # -------------------------
    best_params_file = os.path.join(plots_dir, "best_params.txt")
    with open(best_params_file, "w") as f:
        f.write(f"Best parameters by Validation Accuracy: {best_params_val} | Val Acc = {best_val_acc:.4f}\n")
        f.write(f"Best parameters by Test Accuracy:       {best_params_test} | Test Acc = {best_test_acc:.4f}\n")

    print("\nGrid search complete!\n")
    print(f"Best parameters by Validation Accuracy: {best_params_val} | Val Acc = {best_val_acc:.4f}")
    print(f"Best parameters by Test Accuracy:       {best_params_test} | Test Acc = {best_test_acc:.4f}")

    # -------------------------
    # Plot Test Accuracy vs Parameters
    # -------------------------
    test_acc_plot = [r['test_acc'] for r in results]
    
    # -------------------------
    # 1️ Test Accuracy vs n_estimators
    # -------------------------
    plt.figure(figsize=(8,5))
    for mf in sorted(set([r['max_features'] for r in results])):
        subset = [test_acc_plot[i] for i in range(len(results)) 
                  if results[i]['max_features']==mf and results[i]['min_samples_split']==2]
        plt.plot([50,150,250,350], subset, marker='o', label=f'max_features={mf}')
    plt.xlabel('n_estimators')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs n_estimators')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "rf_test_vs_n_estimators.png"))
    plt.close()
    
    # -------------------------
    # 2️ Test Accuracy vs max_features
    # -------------------------
    plt.figure(figsize=(8,5))
    for n in n_estimators_list:
        subset = [test_acc_plot[i] for i in range(len(results)) 
                  if results[i]['n_estimators']==n and results[i]['min_samples_split']==2]
        plt.plot([0.1,0.3,0.5,0.7,0.9], subset, marker='o', label=f'n_estimators={n}')
    plt.xlabel('max_features')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs max_features')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "rf_test_vs_max_features.png"))
    plt.close()
    
    # -------------------------
    # 3️ Test Accuracy vs min_samples_split
    # -------------------------
    plt.figure(figsize=(8,5))
    for n in n_estimators_list:
        subset = [test_acc_plot[i] for i in range(len(results)) 
                  if results[i]['n_estimators']==n and results[i]['max_features']==0.5]
        plt.plot([2,4,6,8,10], subset, marker='o', label=f'n_estimators={n}')
    plt.xlabel('min_samples_split')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs min_samples_split (max_features=0.5)')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "rf_test_vs_min_samples_split.png"))
    plt.close()

    # Save predictions of best model (validation-based)
    test_predictions = best_model.predict(testX)
    evaluate(test_predictions, testY)
    save_predictions_to_csv(test_predictions, output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage:")
        print(" python3 a.py <train.csv> <validation.csv> <test.csv> <output_file_prefix>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    main(train_path, val_path, test_path, output_path)
