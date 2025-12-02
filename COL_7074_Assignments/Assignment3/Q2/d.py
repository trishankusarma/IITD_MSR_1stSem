import numpy as np
import os
import sys
from neural_network import NeuralNetwork
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
import pandas as pd
import logging
from utils.utils import load_images_from_folders, save_predictions_to_csv, print_metrics

# Ensure directory exists
os.makedirs("outputScripts", exist_ok=True)

# Define log file path
log_file = "outputScripts/d_output.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # overwrite each run
        logging.StreamHandler(sys.stdout)         # show in terminal too
    ]
)

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

    results_c = {}
    avg_f1_scores_test = []
    avg_f1_scores_train = []

    all_predictions = [] # store all predictions

    # Part d: varying depth
    hidden_layers_variants = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]  # varying depths
    
    for layer in hidden_layers_variants:
        logging.info(f"\nTraining Part d model with {layer} layer ...")
        nn = NeuralNetwork(hidden_layers=layer, learning_rate=learning_rate, input_size=n, output_size=r)
        nn.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, use_relu=True)
    
        # Train & Test predictions
        train_preds = nn.predict(train_X, use_relu=True)
        test_preds = nn.predict(test_X, use_relu=True)
    
        # determine correct class range
        num_classes = len(np.unique(train_y))
        labels = np.arange(num_classes)
    
        # Metrics
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            train_y, train_preds, labels=labels, zero_division=0
        )
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            test_y, test_preds, labels=labels, zero_division=0
        )
    
        results_c[str(layer)] = {
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
            f"layer {layer} | Train → Precision: {train_prec_micro:.4f}, Recall: {train_rec_micro:.4f}, F1: {train_f1_micro:.4f} | "
            f"Test → Precision: {test_prec_micro:.4f}, Recall: {test_rec_micro:.4f}, F1: {test_f1_micro:.4f}"
        )
    
        # Macro metrics
        avg_f1_train = f1_score(train_y, train_preds, average="macro")
        avg_f1_test = f1_score(test_y, test_preds, average="macro")
    
        avg_f1_scores_train.append(avg_f1_train)
        avg_f1_scores_test.append(avg_f1_test)
    
        print_metrics(str(layer), results_c[str(layer)])
    
        # save predictions (1-indexed)
        test_preds_1idx = test_preds + 1
        all_predictions.extend(test_preds_1idx)


    # 2 :: Plot the average F1 score vs the number of hidden units
    depth_labels = [str(layers) for layers in hidden_layers_variants]
    
    plt.figure(figsize=(12, 10))
    # Test curve
    plt.plot(depth_labels, avg_f1_scores_test, marker='o', label="Test Macro F1")
    # Train curve
    plt.plot(depth_labels, avg_f1_scores_train, marker='s', linestyle='--', label="Train Macro F1")

    plt.xlabel("Increasing network depth")
    plt.ylabel("Average F1 Score (Macro)")
    plt.title("Average F1 Score vs Depth- With ReLU Activation")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "f1_vs_depth_d.png"))
    plt.close()
    
    # 3:: Save all predictions to CSV
    out_csv_file_path = os.path.join(output_path, "prediction_d.csv")
    save_predictions_to_csv(all_predictions, out_csv_file_path)

    logging.info("Saved predictions to CSV")

if __name__ == "__main__":
    main()