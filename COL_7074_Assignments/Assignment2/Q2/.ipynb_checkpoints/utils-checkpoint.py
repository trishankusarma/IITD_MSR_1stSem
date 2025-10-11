import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_images_from_folder(folderName, label, image_size=(32, 32)):
    images = []
    labels = []
    for filename in os.listdir(folderName):
        if filename.endswith(('.jpg')):
            img_path = os.path.join(folderName, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size)
            img_array = np.array(img, dtype=np.float32) / 255.0  # normalize [0,1]
            img_flat = img_array.flatten()  # 32x32x3 -> 3072
            images.append(img_flat)
            labels.append(label)
    return np.array(images), np.array(labels)


def buildDataset(basePath="", inputClasses=None):
    """
    Builds a dataset DataFrame with flattened normalized image features and labels.
    inputClasses should be a list of class indices (e.g., [4, 5])
    """
    if inputClasses is None or len(inputClasses) < 2:
        raise ValueError("inputClasses must be a list of atleast two class indices.")

    if len(inputClasses) > 2:
        isMultiClassClassification = True
    else:
        isMultiClassClassification = False
    
    data_x = []
    data_y = []

    for idx, class_idx in enumerate(inputClasses):
        class_label = class_names[class_idx]

        if isMultiClassClassification:
            label_value = class_idx
        else:
            label_value = +1 if idx == 0 else -1   # first class → +1, second → -1
            
        X, y = load_images_from_folder(os.path.join(basePath, class_label), label_value)

        print(f"Loaded {class_label} → X: {X.shape}, y: {y.shape}")

        if len(data_x) == 0:
            data_x = X
            data_y = y
        else:
            data_x = np.vstack((data_x, X))
            data_y = np.hstack((data_y, y))

    # Shuffle to mix both classes
    print(f"Suffling to mix both classes")
    data_x, data_y = shuffle(data_x, data_y, random_state=42)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_x)
    df['label'] = data_y

    print(f"\nFinal Dataset Shape {basePath}: {df.shape}")
    return df

def analyze_svm_results(svm, X_train, y_train, X_test, y_test, image_shape=(32, 32, 3)):
    """
    Perform post-training analysis for SVM:
    (a) Support vector stats
    (b) Report w, b, test accuracy
    (c) Visualize top-5 support vectors and w
    """
    print("\n================= SVM Analysis =================")

    # --------------------------
    # (a) Support Vector Stats
    # --------------------------
    num_sv = len(svm.support_vectors)
    percent_sv = (num_sv / len(X_train)) * 100
    print(f"(a) Number of Support Vectors: {num_sv}")
    print(f"    Percentage of Training Samples that constitute the support vectors: {percent_sv:.2f}%")

    # --------------------------
    # (b) w, b and Test Accuracy
    # --------------------------
    print("\n(b) Model Parameters and Test Accuracy:")
    if svm.kernel == 'linear':
        print(f"    Weight Vector (w): shape = {svm.w.shape}")
    else:
        print("    (Gaussian kernel — no explicit weight vector w)")
    
    print(f"    Intercept (b): {svm.bias:.4f}")

    y_pred_test = svm.predict(X_test)
    acc = np.mean(y_pred_test == y_test) * 100
    print(f"    ✅ Test Accuracy: {acc:.2f}%")

    # --------------------------
    # (c) Visualize Support Vectors & w
    # --------------------------
    print("\n(c) Visualization:")

    top5_idx = np.argsort(-svm.alphas)[:5]  # top-5 coefficients
    top5_sv = svm.support_vectors[top5_idx]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        img = top5_sv[i].reshape(image_shape)
        ax.imshow((img - img.min()) / (img.max() - img.min()))
        ax.set_title(f"SV #{i+1}")
        ax.axis("off")
    plt.suptitle("Top-5 Support Vectors (Reshaped as Images)")
    plt.show()

    # Plot weight vector (only for linear kernel)
    if svm.kernel == 'linear' and svm.w is not None:
        w_img = svm.w.reshape(image_shape)
        w_img_norm = (w_img - w_img.min()) / (w_img.max() - w_img.min())
        plt.figure(figsize=(3, 3))
        plt.imshow(w_img_norm)
        plt.title("Weight Vector (w) as Image")
        plt.axis("off")
        plt.show()

    print("================================================\n")
    return acc

def getCountOfIntersectionOfSupportVectors(svm_linear, svm_rbf):
    # support_indices are indices into the training set
    sv_lin_set = set(svm_linear.support_indices.tolist())
    sv_rbf_set = set(svm_rbf.support_indices.tolist())

    num_sv_lin = len(svm_linear.support_vectors)
    num_sv_rbf = len(svm_rbf.support_vectors)
    
    matching_indices = sv_lin_set.intersection(sv_rbf_set)
    num_matching = len(matching_indices)
    perc_matching_of_lin = num_matching / num_sv_lin * 100 if num_sv_lin > 0 else 0
    perc_matching_of_rbf = num_matching / num_sv_rbf * 100 if num_sv_rbf > 0 else 0
    
    print("2(a) SV overlap between linear and RBF:")
    print(f"  Matching support vectors: {num_matching}")
    print(f"  This is {perc_matching_of_lin:.2f}% of linear SVs and {perc_matching_of_rbf:.2f}% of RBF SVs.\n")




