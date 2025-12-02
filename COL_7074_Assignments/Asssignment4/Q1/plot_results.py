import json
import matplotlib.pyplot as plt
import os

#loading data
with open("rnn_training_history.json", "r") as f:
    history = json.load(f)

#extracting lists
train_loss = history["train_loss"]
val_loss = history["val_loss"]
test_loss = history["test_loss"]

train_token = history["train_token_acc"]
val_token = history["val_token_acc"]
test_token = history["test_token_acc"]

train_seq = history["train_seq_acc"]
val_seq = history["val_seq_acc"]
test_seq = history["test_seq_acc"]

train_f1 = history["train_f1"]
val_f1 = history["val_f1"]
test_f1 = history["test_f1"]

epochs = list(range(1, len(train_loss) + 1))

#styling
plt.rcParams.update({"axes.facecolor": "white","figure.facecolor": "white","grid.color": "gray","grid.linestyle": "--","grid.alpha": 0.6,"axes.edgecolor": "black","axes.linewidth": 1.2,
"font.size": 13})

#line colora
colors = {"train": "#1f77b4",  "val":   "#d62728",  "test":  "#2ca02c",  }

line_width = 2.0

os.makedirs("plots", exist_ok=True)

#helper funciton to save plots
def save_single_plot(title, ylabel, train, val, test, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train, label="Train", color=colors["train"], linewidth=line_width)
    plt.plot(epochs, val, label="Validation", color=colors["val"], linewidth=line_width)
    plt.plot(epochs, test, label="Test", color=colors["test"], linewidth=line_width)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()

#here saving the indivisual plots as well
save_single_plot("Loss Curve", "Loss",
                 train_loss, val_loss, test_loss, "loss_curve.png")

save_single_plot("Token Accuracy", "Accuracy",
                 train_token, val_token, test_token, "token_accuracy.png")

save_single_plot("Sequence Accuracy", "Accuracy",
                 train_seq, val_seq, test_seq, "sequence_accuracy.png")

save_single_plot("F1 Score", "F1 Score",
                 train_f1, val_f1, test_f1, "f1_score.png")

print("Saved individual plots in folder: plots/")

plt.figure(figsize=(18, 12))

#Loss Plot
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, color=colors["train"], label="Train", linewidth=line_width)
plt.plot(epochs, val_loss, color=colors["val"], label="Validation", linewidth=line_width)
plt.plot(epochs, test_loss, color=colors["test"], label="Test", linewidth=line_width)
plt.title("Loss Curve", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

#Token Accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, train_token, color=colors["train"], label="Train", linewidth=line_width)
plt.plot(epochs, val_token, color=colors["val"], label="Validation", linewidth=line_width)
plt.plot(epochs, test_token, color=colors["test"], label="Test", linewidth=line_width)
plt.title("Token Accuracy", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

#Sequence Accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, train_seq, color=colors["train"], label="Train", linewidth=line_width)
plt.plot(epochs, val_seq, color=colors["val"], label="Validation", linewidth=line_width)
plt.plot(epochs, test_seq, color=colors["test"], label="Test", linewidth=line_width)
plt.title("Sequence Accuracy", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

#F1 Score
plt.subplot(2, 2, 4)
plt.plot(epochs, train_f1, color=colors["train"], label="Train", linewidth=line_width)
plt.plot(epochs, val_f1, color=colors["val"], label="Validation", linewidth=line_width)
plt.plot(epochs, test_f1, color=colors["test"], label="Test", linewidth=line_width)
plt.title("F1 Score", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/all_plots_combined.png", dpi=300)
plt.close()

print("\nSaved combined plot: plots/all_plots_combined.png\n")
