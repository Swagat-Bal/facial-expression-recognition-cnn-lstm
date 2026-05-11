"""
Dataset preprocessing for folder-based facial expression images.

Expected dataset layout:

face_expression_dataset/
|-- train/
|   |-- angry/
|   |-- disgust/
|   |-- fear/
|   |-- happy/
|   |-- neutral/
|   |-- sad/
|   `-- surprise/
`-- test/
    |-- angry/
    |-- disgust/
    |-- fear/
    |-- happy/
    |-- neutral/
    |-- sad/
    `-- surprise/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "face_expression_dataset"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DISPLAY_NAMES = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetConfig:
    dataset_dir: Path = DEFAULT_DATASET_DIR
    image_size: int = 48
    batch_size: int = 32
    validation_split: float = 0.15
    seed: int = 42


def display_labels() -> list[str]:
    return [DISPLAY_NAMES[name] for name in CLASS_NAMES]


def validate_dataset_dir(dataset_dir: Path) -> None:
    """Validate that the train/test class-folder dataset exists."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    for split in ("train", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing required split folder: {split_dir}")

        missing = [class_name for class_name in CLASS_NAMES if not (split_dir / class_name).exists()]
        if missing:
            raise FileNotFoundError(f"{split_dir} is missing class folders: {missing}")


def count_images(dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    """Count image files in each split/class folder."""
    validate_dataset_dir(dataset_dir)
    counts: Dict[str, Dict[str, int]] = {}

    for split in ("train", "test"):
        counts[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = Path(dataset_dir) / split / class_name
            counts[split][class_name] = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    return counts


def preprocess_image(image: tf.Tensor, config: DatasetConfig, training: bool) -> tf.Tensor:
    """Normalize and augment one grayscale face image."""
    image = tf.cast(image, tf.float32) / 255.0

    if training:
        padded_size = config.image_size + 6
        image = tf.image.resize_with_crop_or_pad(image, padded_size, padded_size)
        image = tf.image.random_crop(image, size=(config.image_size, config.image_size, 1))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def prepare_dataset(
    dataset: tf.data.Dataset,
    config: DatasetConfig,
    training: bool,
) -> tf.data.Dataset:
    """Map raw image/class pairs into batched CNN+LSTM inputs."""
    dataset = dataset.map(
        lambda image, label: (
            preprocess_image(image, config, training),
            tf.one_hot(label, depth=len(CLASS_NAMES)),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.shuffle(4096, seed=config.seed, reshuffle_each_iteration=True)

    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(config: DatasetConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train, validation, and test datasets from image folders."""
    validate_dataset_dir(config.dataset_dir)
    image_size = (config.image_size, config.image_size)
    train_dir = Path(config.dataset_dir) / "train"
    test_dir = Path(config.dataset_dir) / "test"

    train_base = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=None,
        shuffle=True,
        seed=config.seed,
        validation_split=config.validation_split,
        subset="training",
    )
    val_base = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=None,
        shuffle=True,
        seed=config.seed,
        validation_split=config.validation_split,
        subset="validation",
    )
    test_base = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=None,
        shuffle=False,
    )

    return (
        prepare_dataset(train_base, config, training=True),
        prepare_dataset(val_base, config, training=False),
        prepare_dataset(test_base, config, training=False),
    )


def compute_class_weights(dataset_dir: Path, max_weight: float = 3.0) -> Dict[int, float]:
    """Compute capped inverse-frequency class weights from the train split."""
    counts = count_images(dataset_dir)["train"]
    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)

    return {
        index: min(total / (num_classes * max(counts[class_name], 1)), max_weight)
        for index, class_name in enumerate(CLASS_NAMES)
    }


def iter_summary_rows(dataset_dir: Path) -> Iterable[tuple[str, str, int]]:
    counts = count_images(dataset_dir)
    for split, split_counts in counts.items():
        for class_name in CLASS_NAMES:
            yield split, DISPLAY_NAMES[class_name], split_counts[class_name]


def print_summary(dataset_dir: Path) -> None:
    print(f"Dataset: {Path(dataset_dir).resolve()}")
    print("\nSplit      Class       Images")
    print("-" * 32)
    for split, label, total in iter_summary_rows(dataset_dir):
        print(f"{split:<10} {label:<10} {total:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the face expression dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    print_summary(args.dataset)


if __name__ == "__main__":
    main()
