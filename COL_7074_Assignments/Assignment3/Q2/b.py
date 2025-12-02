import numpy as np
import os
import sys
from neural_network import NeuralNetwork
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
import pandas as pd
import logging
from utils.utils import load_images_from_folders, save_predictions_to_csv, print_metrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    logging.info("Loading training data...")
    train_X, train_y = load_images_from_folders(train_path)
    logging.info(f"Training samples: {train_X.shape[0]}")

    logging.info("Loading test data...")
    test_X, test_y = load_images_from_folders(test_path)
    logging.info(f"Test samples: {test_X.shape[0]}")

    M = 32
    n = 32 * 32 * 3
    r = 36
    learning_rate = 0.01
    epochs = 400
    batch_size = M

    results_b = {}
    avg_f1_scores_test = []
    avg_f1_scores_train = []

    all_predictions = []  # store all predictions

    # Part b: single hidden layer, varying neurons
    hidden_units_b = [1, 5, 10, 50, 100]

    for units in hidden_units_b:
        logging.info(f"\nTraining Part b model with {units} hidden units ...")

        nn = NeuralNetwork(hidden_layers=[units],
                           learning_rate=learning_rate,
                           input_size=n,
                           output_size=r)

        nn.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, use_relu=False)

        # Predictions
        train_preds = nn.predict(train_X)
        test_preds = nn.predict(test_X)

        # Per-class metrics
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            train_y, train_preds, labels=np.arange(0, r), zero_division=0
        )
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            test_y, test_preds, labels=np.arange(0, r), zero_division=0
        )

        results_b[str(units)] = {
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1
        }

        # Micro metrics
        train_f1_micro = f1_score(train_y, train_preds, average="micro")
        test_f1_micro = f1_score(test_y, test_preds, average="micro")

        logging.info(
            f"Units {units} | Train F1_micro: {train_f1_micro:.4f} | "
            f"Test F1_micro: {test_f1_micro:.4f}"
        )

        avg_f1_test = f1_score(test_y, test_preds, average="macro")
        avg_f1_scores_test.append(avg_f1_test)

        avg_f1_train = f1_score(train_y, train_preds, average="macro")
        avg_f1_scores_train.append(avg_f1_train)

        print_metrics(str(units), results_b[str(units)])

        # Store predictions (1-indexed)
        test_preds = test_preds + 1
        all_predictions.extend(test_preds)

    # Plot Macro F1 vs hidden units
    plt.figure(figsize=(8, 6))

    plt.plot(hidden_units_b, avg_f1_scores_test, marker='o', label="Test Macro F1")
    plt.plot(hidden_units_b, avg_f1_scores_train, marker='s', linestyle='--', label="Train Macro F1")

    # Annotate test points
    for x, y in zip(hidden_units_b, avg_f1_scores_test):
        plt.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8)

    # Annotate train points
    for x, y in zip(hidden_units_b, avg_f1_scores_train):
        plt.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points", xytext=(0, -12),
                     ha='center', fontsize=8)

    plt.xlabel("Number of Hidden Units", fontsize=10)
    plt.ylabel("Average F1 Score (Macro)", fontsize=10)
    plt.title("Average F1 Score vs Hidden Units (Single Layer)", fontsize=11)

    plt.xticks(hidden_units_b, rotation=30)
    plt.grid(True, linewidth=0.4)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "f1_vs_hidden_units_b.png"))
    plt.close()

    # Save predictions
    out_csv = os.path.join(output_path, "prediction_b.csv")
    save_predictions_to_csv(all_predictions, out_csv)
    logging.info("Saved predictions to CSV")

if __name__ == "__main__":
    main()
