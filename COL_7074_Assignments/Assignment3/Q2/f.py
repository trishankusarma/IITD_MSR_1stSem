import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils.utils import load_images_from_folders, plotF1Scores, save_predictions_to_csv
from neural_network import NeuralNetwork
import sys
import os
import pandas as pd

np.seterr(over='ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def compute_f1(model, X, y, use_relu=True):
    preds = model.predict(X, use_relu)
    return f1_score(y, preds, average="macro")


def main():
    # Read command line arguments
    train_path_digits = sys.argv[1]
    test_path_digits = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)

    X_digits_train, y_digits_train = load_images_from_folders(train_path_digits)
    logging.info(f"Training samples: {X_digits_train.shape[0]}")
    X_digits_test, y_digits_test = load_images_from_folders(test_path_digits)
    logging.info(f"Testing samples: {X_digits_test.shape[0]}")

    input_size = X_digits_train.shape[1]
    hidden_layers = [512, 256, 128, 64]
    learning_rate = 0.01
    epochs = 20
    batch_size = 32

    all_predictions = []

    # PART 1 — Train from scratch
    logging.info("Training digits model from scratch...")
    scratch_model = NeuralNetwork(hidden_layers=hidden_layers,
                                  output_size=10,
                                  learning_rate=learning_rate,
                                  input_size=input_size)

    scratch_train_f1s = []
    scratch_test_f1s = []

    for epoch in range(epochs):
        scratch_model.fit(X_digits_train, y_digits_train, epochs=1, batch_size=batch_size, use_relu=True)
        train_f1 = compute_f1(scratch_model, X_digits_train, y_digits_train)
        test_f1 = compute_f1(scratch_model, X_digits_test, y_digits_test)

        scratch_train_f1s.append(train_f1)
        scratch_test_f1s.append(test_f1)

        logging.info(f"[Scratch] Epoch {epoch+1}/{epochs} | Train F1={train_f1:.4f} | Test F1={test_f1:.4f}")

    epochList = range(1, epochs+1)

    plotF1Scores(epochList, scratch_test_f1s, scratch_train_f1s,
                 labelTest="Scratch Test F1",
                 labelTrain="Scratch Train F1",
                 xLabel="Epoch", yLabel="Macro F1 Score",
                 title="Digits Classification: Training from Scratch",
                 outputFile=f"{output_path}/digits_scratch_f1.png")

    # Store predictions (+1 because labels expected 1–10)
    y_pred_scratch_digits_plus1 = scratch_model.predict(X_digits_test, use_relu = True) + 1
    all_predictions.extend(y_pred_scratch_digits_plus1)

    # PART 2 — Transfer Learning
    data = np.load("consonant_model_weights.npz")
    consonant_params = {key: data[key] for key in data.files}

    
    for k, v in consonant_params.items():
        print(k, v.shape)

    transfer_model = NeuralNetwork(hidden_layers=hidden_layers,
                                   output_size=10,
                                   learning_rate=learning_rate,
                                   input_size=input_size)

    # Copy all layers except output
    for i in range(transfer_model.L - 1):
        transfer_model.params[f"W{i}"] = consonant_params[f"W{i}"].copy()
        transfer_model.params[f"b{i}"] = consonant_params[f"b{i}"].copy()

    transfer_train_f1s = []
    transfer_test_f1s = []

    logging.info("Fine-tuning transfer model on digits dataset...")
    for epoch in range(epochs):
        transfer_model.fit(X_digits_train, y_digits_train, epochs=1, batch_size=batch_size, use_relu=True)
        train_f1 = compute_f1(transfer_model, X_digits_train, y_digits_train)
        test_f1 = compute_f1(transfer_model, X_digits_test, y_digits_test)

        transfer_train_f1s.append(train_f1)
        transfer_test_f1s.append(test_f1)

        logging.info(f"[Transfer] Epoch {epoch+1}/{epochs} | Train F1={train_f1:.4f} | Test F1={test_f1:.4f}")

    plotF1Scores(epochList, transfer_test_f1s, transfer_train_f1s,
                 labelTest="Transfer Model Test F1",
                 labelTrain="Transfer Model Train F1",
                 xLabel="Epoch", yLabel="Macro F1 Score",
                 title="Digits Classification: Transfer Model Learning",
                 outputFile=f"{output_path}/digits_transfer_f2.png")

    # Combine both plots
    plt.figure(figsize=(9, 6))
    plt.plot(epochList, scratch_test_f1s, marker='o', label="Scratch Test F1")
    plt.plot(epochList, transfer_test_f1s, marker='s', label="Transfer Test F1")

    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 Score")
    plt.title("Digits Classification: Scratch vs Transfer Learning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/digits_scratch_vs_transfer.png")
    plt.close()

    logging.info("Comparison plot saved successfully.")

    # Predictions (+1)
    y_pred_transfer_digits_plus1 = transfer_model.predict(X_digits_test, use_relu = True) + 1
    all_predictions.extend(y_pred_transfer_digits_plus1)

    # Save final predictions to CSV
    out_csv_file_path = os.path.join(output_path, "prediction_f.csv")
    save_predictions_to_csv(all_predictions, out_csv_file_path)

    logging.info("Saved predictions to CSV")


if __name__ == "__main__":
    main()
