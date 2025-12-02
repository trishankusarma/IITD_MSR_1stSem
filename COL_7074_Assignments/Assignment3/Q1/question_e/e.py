#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import sys
import os

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from utils import pre_process, save_predictions_to_csv, evaluate

def main(train_path, val_path, test_path, output_path):

    # ---------------------- Load Data -----------------------
    print("\nLoading data ...")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}")

    # ---------------------- Preprocess ----------------------
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]
    categorical_cols = ["team", "opp", "host", "month"]

    print("\nPreprocessing ...")
    X_train, y_train, X_val, y_val, X_test, y_test = pre_process(
        train, val, test,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols
    )

    # ---------------------- Part (i) Varying max_depth -----------------------
    depths = [15, 25, 35, 45]
    train_acc, val_acc, test_acc = [], [], []

    print("\n=== Running DecisionTreeClassifier: Varying max_depth ===")
    best_depth_model = None
    best_depth_val_score = -1
    best_depth = None

    for d in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d, random_state=0)
        clf.fit(X_train, y_train)

        tr = accuracy_score(y_train, clf.predict(X_train))
        va = accuracy_score(y_val, clf.predict(X_val))
        te = accuracy_score(y_test, clf.predict(X_test))

        train_acc.append(tr)
        val_acc.append(va)
        test_acc.append(te)

        print(f"[{datetime.now()}] Depth={d} | Train={tr:.4f} | Val={va:.4f} | Test={te:.4f}")

        if va > best_depth_val_score:
            best_depth_val_score = va
            best_depth_model = clf
            best_depth = d

    # Plot
    plt.figure()
    plt.plot(depths, train_acc, label='Train')
    plt.plot(depths, val_acc, label='Validation')
    plt.plot(depths, test_acc, label='Test')
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Max Depth")
    plt.legend()
    plt.savefig(f"plots/accuracy_vs_depth.png")
    plt.close()

    print(f"\nBest depth based on validation accuracy: {best_depth} (Val Acc={best_depth_val_score:.4f})")


    # ---------------------- Part (ii) Varying ccp_alpha -----------------------
    alphas = [0.0, 0.0001, 0.0003, 0.0005]
    train_acc_a, val_acc_a, test_acc_a = [], [], []

    print("\n=== Running DecisionTreeClassifier: Varying ccp_alpha ===")
    best_alpha_model = None
    best_alpha_val_score = -1
    best_alpha = None

    for alpha in alphas:
        clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=0)
        clf.fit(X_train, y_train)

        tr = accuracy_score(y_train, clf.predict(X_train))
        va = accuracy_score(y_val, clf.predict(X_val))
        te = accuracy_score(y_test, clf.predict(X_test))

        train_acc_a.append(tr)
        val_acc_a.append(va)
        test_acc_a.append(te)

        print(f"[{datetime.now()}] Alpha={alpha} | Train={tr:.4f} | Val={va:.4f} | Test={te:.4f}")

        if va > best_alpha_val_score:
            best_alpha_val_score = va
            best_alpha_model = clf
            best_alpha = alpha

    # Plot
    plt.figure()
    plt.plot(alphas, train_acc_a, label='Train')
    plt.plot(alphas, val_acc_a, label='Validation')
    plt.plot(alphas, test_acc_a, label='Test')
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs ccp_alpha")
    plt.legend()
    plt.savefig(f"plots/accuracy_vs_alpha.png")
    plt.close()

    print(f"\nBest alpha based on validation accuracy: {best_alpha} (Val Acc={best_alpha_val_score:.4f})")

    # ---------------------- (e) Final Best Tree -----------------------

    print("\n=== (e) Selecting BEST MODEL based on validation accuracy ===")

    if best_depth_val_score >= best_alpha_val_score:
        best_model = best_depth_model
        best_type = f"max_depth={best_depth}"
        best_val = best_depth_val_score
    else:
        best_model = best_alpha_model
        best_type = f"ccp_alpha={best_alpha}"
        best_val = best_alpha_val_score

    print(f"\nBest Model = {best_type} with Validation Accuracy = {best_val:.4f}")

    best_predictY_test = best_model.predict(X_test)
    evaluate(best_predictY_test, y_test)
    save_predictions_to_csv(best_predictY_test, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 question_e.py <train.csv> <validation.csv> <test.csv> <output_dir>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    main(train_path, val_path, test_path, output_path)
