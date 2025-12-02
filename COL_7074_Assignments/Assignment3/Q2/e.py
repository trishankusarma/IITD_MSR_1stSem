import numpy as np
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score
)

from utils.utils import load_images_from_folders, print_metrics, save_predictions_to_csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)

    logging.info("Loading training data...")
    train_X, train_y = load_images_from_folders(train_path)
    logging.info(f"Training samples: {train_X.shape[0]}")

    logging.info("Loading test data...")
    test_X, test_y = load_images_from_folders(test_path)
    logging.info(f"Test samples: {test_X.shape[0]}")

    # Architectures same as Part (c)
    hidden_layers_variants = [
        (512,),
        (512, 256),
        (512, 256, 128),
        (512, 256, 128, 64)
    ]

    learning_rate = 0.01
    batch_size = 32
    results_e = {}
    avg_f1_scores_test = []
    avg_f1_scores_train = []
    all_predictions = []

    for layer in hidden_layers_variants:
        logging.info(f"\nTraining sklearn MLP with architecture: {layer}")

        clf = MLPClassifier(
            hidden_layer_sizes=layer,
            activation="relu",
            solver="sgd",
            alpha=0,
            batch_size=batch_size,
            learning_rate="constant",
            learning_rate_init=learning_rate,
            max_iter=400,              # stopping criteria
            shuffle=True,
            verbose=False
        )

        clf.fit(train_X, train_y)

        train_preds = clf.predict(train_X)
        test_preds = clf.predict(test_X)

        # Per-class metrics
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            train_y, train_preds, labels=np.arange(0, 36), zero_division=0
        )
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            test_y, test_preds, labels=np.arange(0, 36), zero_division=0
        )

        results_e[str(layer)] = {
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1
        }

        # Micro metrics
        train_prec_micro = precision_score(train_y, train_preds, average="micro", zero_division=0)
        train_rec_micro = recall_score(train_y, train_preds, average="micro", zero_division=0)
        train_f1_micro = f1_score(train_y, train_preds, average="micro", zero_division=0)

        test_prec_micro = precision_score(test_y, test_preds, average="micro", zero_division=0)
        test_rec_micro = recall_score(test_y, test_preds, average="micro", zero_division=0)
        test_f1_micro = f1_score(test_y, test_preds, average="micro", zero_division=0)

        logging.info(
            f"Architecture {layer} | Train → Precision: {train_prec_micro:.4f}, "
            f"Recall: {train_rec_micro:.4f}, F1: {train_f1_micro:.4f} | "
            f"Test → Precision: {test_prec_micro:.4f}, Recall: {test_rec_micro:.4f}, "
            f"F1: {test_f1_micro:.4f}"
        )

        # Macro F1 tracking
        avg_f1_train = f1_score(train_y, train_preds, average="macro")
        avg_f1_test = f1_score(test_y, test_preds, average="macro")

        avg_f1_scores_train.append(avg_f1_train)
        avg_f1_scores_test.append(avg_f1_test)

        logging.info(f"Layer {layer} → Train Macro-F1 = {avg_f1_train:.4f}")
        logging.info(f"Layer {layer} → Test Macro-F1 = {avg_f1_test:.4f}")

        print_metrics(str(layer), results_e[str(layer)])

        # Save predictions (+1 indexing)
        predictions = test_preds + 1
        all_predictions.extend(predictions)

    # Plot macro F1 vs depth
    depth_labels = [str(layers) for layers in hidden_layers_variants]

    plt.figure(figsize=(8, 6))
    plt.plot(depth_labels, avg_f1_scores_test, marker='o', label="Test Macro F1")
    plt.plot(depth_labels, avg_f1_scores_train, marker='s', linestyle='--', label="Train Macro F1")

    # Annotate test points
    for x, y in zip(depth_labels, avg_f1_scores_test):
        plt.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8)
    
    # Annotate train points
    for x, y in zip(depth_labels, avg_f1_scores_train):
        plt.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points", xytext=(0, -12),
                     ha='center', fontsize=8)

    plt.xlabel("Network Depth (Hidden Layer Structure)", fontsize=10)
    plt.ylabel("Average Macro F1 Score", fontsize=10)
    plt.title("Part (e): F1 Score vs Depth using sklearn MLP", fontsize=11)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "f1_vs_depth_e.png"))
    plt.close()

    # Save final predictions to CSV
    out_csv_file_path = os.path.join(output_path, "prediction_e.csv")
    save_predictions_to_csv(all_predictions, out_csv_file_path)

    logging.info("Saved predictions to CSV")


if __name__ == "__main__":
    main()
