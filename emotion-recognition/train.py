"""
Train a CNN + LSTM classifier on the provided face_expression_dataset folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import (
    CLASS_NAMES,
    DEFAULT_DATASET_DIR,
    DatasetConfig,
    compute_class_weights,
    display_labels,
    build_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_gpu() -> None:
    """Use CUDA automatically when TensorFlow can see a compatible GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No TensorFlow GPU detected. Training will run on CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not enable memory growth for {gpu}: {exc}")
    print(f"TensorFlow GPU enabled: {len(gpus)} device(s).")


def build_cnn_lstm_model(image_size: int, num_classes: int) -> tf.keras.Model:
    """Build a stronger CNN + LSTM model for static expression images.

    The CNN produces a spatial feature map, then the feature map is reshaped into
    a patch sequence for a bidirectional LSTM. This is a better fit for your
    still-image dataset than repeating the same image across fake timesteps.
    """
    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name="face_image")
    regularizer = tf.keras.regularizers.l2(1e-4)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    spatial_steps = (image_size // 8) * (image_size // 8)
    x = tf.keras.layers.Reshape((spatial_steps, 256), name="spatial_feature_sequence")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.25))(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_expression_classifier")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def save_history_plot(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history_dict.get("accuracy", []), label="Train")
    axes[0].plot(epochs, history_dict.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history_dict.get("loss", []), label="Train")
    axes[1].plot(epochs, history_dict.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    output_dir: Path,
) -> Tuple[float, np.ndarray]:
    """Evaluate on the test split and save a confusion matrix."""
    y_true = []
    y_pred = []

    for batch_x, batch_y in test_dataset:
        probabilities = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))
        y_pred.extend(np.argmax(probabilities, axis=1))

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    labels = np.arange(len(CLASS_NAMES))
    label_names = display_labels()

    accuracy = accuracy_score(y_true_array, y_pred_array)
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=labels)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true_array,
            y_pred_array,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        )
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=labels,
        yticks=labels,
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, matrix[row, col], ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    return accuracy, matrix


def save_metadata(output_dir: Path, config: DatasetConfig, accuracy: float) -> None:
    metadata = {
        "class_names": CLASS_NAMES,
        "display_labels": display_labels(),
        "image_size": config.image_size,
        "architecture": "cnn_spatial_features_plus_bidirectional_lstm",
        "test_accuracy": float(accuracy),
        "dataset_dir": str(Path(config.dataset_dir).resolve()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_history_json(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (output_dir / "history.json").write_text(
        json.dumps(serializable_history, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN + LSTM facial expression classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()

    configure_gpu()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        dataset_dir=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    train_dataset, val_dataset, test_dataset = build_datasets(config)
    class_weights = None if args.no_class_weights else compute_class_weights(
        config.dataset_dir,
        max_weight=args.max_class_weight,
    )
    print(f"Class weights: {class_weights if class_weights else 'disabled'}")

    model = build_cnn_lstm_model(
        image_size=config.image_size,
        num_classes=len(CLASS_NAMES),
    )
    model.summary()

    checkpoint_path = args.model_dir / "best_cnn_lstm_expression_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=7,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    final_model_path = args.model_dir / "cnn_lstm_expression_model.keras"
    model.save(final_model_path)
    save_history_json(history, args.model_dir)
    save_history_plot(history, args.model_dir)

    accuracy, _ = evaluate_model(model, test_dataset, args.model_dir)
    save_metadata(args.model_dir, config, accuracy)

    print(f"Saved best model: {checkpoint_path}")
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
