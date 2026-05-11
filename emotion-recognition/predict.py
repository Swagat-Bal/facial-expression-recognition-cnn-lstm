"""
Single-image prediction utilities for the trained CNN + LSTM model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from PIL import Image

from preprocess import CLASS_NAMES, DISPLAY_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_cnn_lstm_expression_model.keras"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]


def load_expression_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with python train.py")
    return tf.keras.models.load_model(model_path)


def pil_to_model_input(
    image: Image.Image,
    image_size: int = 48,
    sequence_length: int | None = None,
    sequence_model: bool = False,
) -> np.ndarray:
    """Convert one PIL image to a model input batch."""
    gray = image.convert("L").resize((image_size, image_size))
    array = np.asarray(gray, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=-1)

    if sequence_model:
        steps = sequence_length or 6
        sequence = np.repeat(array[np.newaxis, ...], steps, axis=0)
        return np.expand_dims(sequence, axis=0)

    return np.expand_dims(array, axis=0)


def predict_pil_image(
    model: tf.keras.Model,
    image: Image.Image,
    image_size: int = 48,
    sequence_length: int | None = None,
) -> Prediction:
    sequence_model = len(model.input_shape) == 5
    batch = pil_to_model_input(
        image,
        image_size=image_size,
        sequence_length=sequence_length,
        sequence_model=sequence_model,
    )
    probabilities_array = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(probabilities_array))

    probabilities = {
        DISPLAY_NAMES[class_name]: float(probabilities_array[index])
        for index, class_name in enumerate(CLASS_NAMES)
    }
    label = DISPLAY_NAMES[CLASS_NAMES[best_index]]
    return Prediction(
        label=label,
        confidence=float(probabilities_array[best_index]),
        probabilities=probabilities,
    )


def predict_image_file(
    image_path: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    image_size: int = 48,
    sequence_length: int | None = None,
) -> Prediction:
    model = load_expression_model(model_path)
    image = Image.open(image_path)
    return predict_pil_image(model, image, image_size=image_size, sequence_length=sequence_length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict expression for a single face image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=None, help="Only used with older sequence-input models.")
    args = parser.parse_args()

    result = predict_image_file(
        image_path=args.image,
        model_path=args.model,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
    )

    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print("\nProbabilities:")
    for label, probability in sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"{label:<10} {probability * 100:6.2f}%")


if __name__ == "__main__":
    main()
