#!/usr/bin/env python3
"""
ECG Arrhythmia Detector — Model Training & Evaluation
======================================================
Builds a 1D Convolutional Neural Network (CNN) to classify individual
heartbeats into 5 AAMI classes, trains it on segmented MIT-BIH data,
evaluates on a held-out test set, and saves the model + metrics + plots
for the Streamlit dashboard.

The CNN architecture is designed for 1D signal classification:
- Conv1D layers act as learned feature detectors that slide across the
  250-sample ECG waveform, automatically learning to detect patterns
  like sharp peaks (QRS complex), wide deflections (PVC morphology),
  and irregular spacing
- MaxPooling reduces dimensionality and provides translation invariance
- Dense layers combine the extracted features for classification

Usage:
    python train_model.py          (requires data_prep.py to have been run first)
"""

import os
import json
import numpy as np

# Use non-interactive matplotlib backend (we're saving plots to disk, not displaying)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Configuration ─────────────────────────────────────────────────────

OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ecg_model.keras")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CM_IMAGE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
BEAT_LEN = 250  # samples per beat (must match data_prep.py)

# ── Training Hyperparameters ──────────────────────────────────────────
# These are tuned for the MIT-BIH dataset at hackathon scale:
# - 20 epochs is usually enough to converge on this dataset
# - Batch size 128 balances speed vs. gradient stability
# - Adam at 0.001 is the standard starting point
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# ── Focal Loss (for class imbalance) ──────────────────────────────────

def focal_loss(gamma=2.0):
    """
    Focal Loss for multi-class classification.

    Focal loss automatically down-weights well-classified examples and focuses
    training on hard, misclassified ones. This is more effective than class
    weights for imbalanced datasets because it adapts per-sample rather than
    per-class.

    gamma=2.0 means: if the model is 90% confident on a sample, that sample's
    loss contribution is reduced by (1-0.9)^2 = 0.01x. Hard samples where
    the model is only 30% confident contribute (1-0.3)^2 = 0.49x — nearly
    full weight.
    """
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Convert sparse labels to one-hot
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        # Standard cross-entropy per class
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        # Focal modulation: down-weight easy examples
        weight = y_true_onehot * tf.pow(1 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_sum(focal, axis=-1)
    return focal_loss_fn


# ── Model Architecture ───────────────────────────────────────────────

def residual_block(x, filters, kernel_size=5):
    """
    A residual block with skip connection.

    The skip connection allows gradients to flow directly through the network,
    enabling deeper architectures without vanishing gradient problems.
    If input channels != output filters, a 1x1 conv adjusts the dimensions.
    """
    shortcut = x

    # Main path: two conv layers with batch normalization
    out = layers.Conv1D(filters, kernel_size, padding="same")(x)
    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    out = layers.Conv1D(filters, kernel_size, padding="same")(out)
    out = layers.BatchNormalization()(out)

    # Adjust shortcut dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection and activate
    out = layers.Add()([out, shortcut])
    out = layers.Activation("relu")(out)
    return out


def build_model(input_length, num_classes):
    """
    Build a Residual 1D-CNN for ECG beat classification.

    Architecture:
      - Initial Conv1D(64, 7) for broad feature extraction
      - 3 Residual Blocks with increasing filters (64 -> 128 -> 256)
        with skip connections for gradient flow
      - GlobalAveragePooling for translation invariance
      - Dense classification head with dropout regularization

    Uses Focal Loss instead of cross-entropy for better handling
    of class imbalance.
    """
    inputs = layers.Input(shape=(input_length, 1))

    # Initial broad feature extraction
    x = layers.Conv1D(64, kernel_size=7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    # Output: (125, 64)

    # Residual Block 1: local pattern detection
    x = residual_block(x, 64, kernel_size=5)
    x = layers.MaxPooling1D(pool_size=2)(x)
    # Output: (62, 64)

    # Residual Block 2: morphological pattern detection
    x = residual_block(x, 128, kernel_size=5)
    x = layers.MaxPooling1D(pool_size=2)(x)
    # Output: (31, 128)

    # Residual Block 3: high-level shape detection
    x = residual_block(x, 256, kernel_size=3)
    x = layers.MaxPooling1D(pool_size=2)(x)
    # Output: (15, 256)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0),
        metrics=["accuracy"],
    )

    return model


# ── Visualization ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Generate and save a confusion matrix heatmap.

    Shows both raw counts and percentages (normalized by true class)
    so viewers can assess both absolute numbers and rates.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize to row percentages (what fraction of each true class
    # was predicted as each other class)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for classes with no samples
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = cm.astype("float") / row_sums * 100

    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (% of true class)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, label="Percentage")

    # Axis labels with sample counts
    tick_marks = np.arange(len(class_names))
    class_labels_y = [f"{c}\n(n={cm[i].sum()})" for i, c in enumerate(class_names)]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_labels_y, fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)

    # Annotate each cell with count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def plot_training_history(history, save_path):
    """Plot training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history["loss"], label="Train", linewidth=2)
    ax2.plot(history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Model Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history plot saved to: {save_path}")


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ECG Arrhythmia Detector - Model Training & Evaluation")
    print("=" * 65)

    # ── Step 1: Load prepared data ────────────────────────────────────
    print("\n[1/5] Loading prepared data...")
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    print(f"  Training set : {X_train.shape[0]:>6,} beats | shape: {X_train.shape}")
    print(f"  Test set     : {X_test.shape[0]:>6,} beats | shape: {X_test.shape}")

    # Class distribution summary
    for name, y in [("Train", y_train), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  {name} distribution: {dist}")

    # ── Step 2: Oversample + Augment minority classes ─────────────────
    # The dataset is heavily imbalanced. Random oversampling brings minority
    # classes up to a reasonable fraction of the majority class, then data
    # augmentation creates realistic variants to prevent overfitting.
    print("\n[2/5] Oversampling minority classes + data augmentation...")

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_count = max(class_counts)
    # Target: bring every class to at least 50% of the majority class count
    target_min = int(max_count * 0.5)

    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for cls, count in zip(unique_classes, class_counts):
        if count < target_min:
            # Number of new samples to generate
            n_new = target_min - count
            cls_indices = np.where(y_train == cls)[0]

            for _ in range(n_new):
                # Pick a random sample from this class
                idx = np.random.choice(cls_indices)
                beat = X_train[idx].copy()

                # Apply random augmentations:
                # 1. Gaussian noise (simulates electrode noise)
                noise = np.random.normal(0, 0.02, beat.shape)
                beat = beat + noise

                # 2. Amplitude scaling (simulates signal strength variation)
                scale = np.random.uniform(0.85, 1.15)
                beat = beat * scale

                # 3. Time shift (simulates slight R-peak alignment variation)
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    beat = np.roll(beat, shift)

                # Clip to [0, 1] range
                beat = np.clip(beat, 0, 1)

                X_aug_list.append(beat.reshape(1, -1))
                y_aug_list.append(np.array([cls]))

    X_train = np.concatenate(X_aug_list, axis=0)
    y_train = np.concatenate(y_aug_list, axis=0)

    # Shuffle augmented data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    print(f"  After augmentation: {X_train.shape[0]:,} training beats")
    unique_aug, counts_aug = np.unique(y_train, return_counts=True)
    aug_dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique_aug, counts_aug)}
    print(f"  New distribution: {aug_dist}")

    # Reshape for Conv1D: (num_beats, 250) -> (num_beats, 250, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # ── Step 3: Build and train Residual CNN ────────────────────────────
    # Focal loss replaces class weights — it automatically focuses on
    # hard-to-classify samples without destabilizing training.
    print("\n[3/5] Building and training Residual 1D-CNN with Focal Loss...")
    model = build_model(X_train.shape[1], len(CLASS_NAMES))
    model.summary()

    # Callbacks for optimal training:
    # - ReduceLROnPlateau: reduces LR when val_accuracy plateaus
    # - EarlyStopping: stops when val_accuracy stagnates, restores best
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1, mode="max"
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model to disk
    model.save(MODEL_PATH)
    print(f"\n  [OK] Model saved to: {MODEL_PATH}")

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist_dict, f, indent=2)

    # Plot training curves
    plot_training_history(
        history.history,
        os.path.join(OUTPUT_DIR, "training_history.png")
    )

    # ── Step 4: Evaluate on held-out test set ─────────────────────────
    print("\n[4/5] Evaluating on held-out test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  +-------------------------------------------+")
    print(f"  |  Overall Accuracy: {accuracy*100:5.1f}%                |")
    print(f"  +-------------------------------------------+")

    # Detailed classification report
    report_str = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(f"\n  Per-Class Classification Report:\n")
    for line in report_str.split("\n"):
        print(f"  {line}")

    # Save text report
    with open(REPORT_PATH, "w") as f:
        f.write("ECG Arrhythmia Detector - Classification Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report_str)
    print(f"\n  Report saved to: {REPORT_PATH}")

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASS_NAMES)))
    plot_confusion_matrix(cm, CLASS_NAMES, CM_IMAGE_PATH)

    # ── Step 5: Save metrics as JSON for dashboard ────────────────────
    print("\n[5/5] Saving metrics for dashboard...")
    report_dict = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "epochs_trained": len(history.history["loss"]),
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    for cls_name in CLASS_NAMES:
        if cls_name in report_dict:
            metrics["per_class"][cls_name] = {
                "precision": round(report_dict[cls_name]["precision"], 4),
                "recall": round(report_dict[cls_name]["recall"], 4),
                "f1_score": round(report_dict[cls_name]["f1-score"], 4),
                "support": int(report_dict[cls_name]["support"]),
            }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved to: {METRICS_PATH}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Training & Evaluation Complete!")
    print(f"")
    print(f"  Saved files:")
    print(f"    Model            : {MODEL_PATH}")
    print(f"    Metrics (JSON)   : {METRICS_PATH}")
    print(f"    Confusion matrix : {CM_IMAGE_PATH}")
    print(f"    Report (text)    : {REPORT_PATH}")
    print(f"    Training curves  : {os.path.join(OUTPUT_DIR, 'training_history.png')}")
    print(f"")
    print(f"  Key results:")
    print(f"    Overall accuracy : {accuracy*100:.1f}%")
    for cls_name in CLASS_NAMES:
        if cls_name in metrics["per_class"]:
            m = metrics["per_class"][cls_name]
            print(f"    {cls_name} recall       : {m['recall']*100:.1f}% "
                  f"(precision: {m['precision']*100:.1f}%, "
                  f"support: {m['support']})")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
