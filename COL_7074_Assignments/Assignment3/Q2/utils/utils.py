import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import logging

# ---------------------------
# Load Images
# ---------------------------
def load_images_from_folders(data_path, image_size=(32, 32)):
    X = []
    y = []
    class_folders = sorted(os.listdir(data_path))

    for label, folder_name in enumerate(class_folders):
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                img = cv2.imread(file_path)

                if img is None:
                    logging.info(f"Error loading {file_path}: Not an image or unreadable")
                    continue

                # Convert BGR → RGB (to match your previous pipeline)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

                # Convert to vector
                img_vector = img.flatten()

                X.append(img_vector)
                y.append(label)

            except Exception as e:
                logging.info(f"Error loading {file_path}: {e}")

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)

    return X, y

def print_metrics(arch, res):
    print(f"Model: {arch}")

    print("----Train----")
    for i in range(36):
        print(f"Class {i+1:2d} → "
              f"P={res['train_precision'][i]:.3f}, "
              f"R={res['train_recall'][i]:.3f}, "
              f"F1={res['train_f1'][i]:.3f}")

    print("----Test----")
    for i in range(36):
        print(f"Class {i+1:2d} → "
              f"P={res['test_precision'][i]:.3f}, "
              f"R={res['test_recall'][i]:.3f}, "
              f"F1={res['test_f1'][i]:.3f}")

# ---------------------------
# Save Predictions
# ---------------------------
def save_predictions_to_csv(pred_list, csv_path):
    df = pd.DataFrame({"prediction": pred_list})
    df.to_csv(csv_path, index=False)
    print(f" Saved predictions to {csv_path}")

def plotF1Scores(epochList, scratch_test_f1s, scratch_train_f1s, labelTest = "Scratch Test F1", labelTrain = "Scratch Train F1", xLabel = "", yLabel = "", title = "", outputFile = ""):
    # Plot Test F1 over epochs
    plt.figure(figsize=(8,6))
    plt.plot(epochList, scratch_test_f1s, marker='o', label=labelTest)
    plt.plot(epochList, scratch_train_f1s, marker='s', linestyle='--', label=labelTrain)
    # Annotate test points
    for x, y in zip(epochList, scratch_test_f1s):
        plt.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=6)
    
    # Annotate train points
    for x, y in zip(epochList, scratch_train_f1s):
        plt.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points", xytext=(0, -12),
                     ha='center', fontsize=6)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputFile)
    plt.close()