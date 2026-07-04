from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Alzheimer MRI classifier.")
    parser.add_argument(
        "--dataset-dir",
        default="dataset/OriginalDataset",
        help="Directory containing one subfolder per Alzheimer class.",
    )
    parser.add_argument(
        "--model",
        choices=("cnn", "mobilenet"),
        default="cnn",
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Expected folders: "
            + ", ".join(str(dataset_dir / name) for name in CLASS_NAMES)
        )

    missing = [name for name in CLASS_NAMES if not (dataset_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders under {dataset_dir}: {', '.join(missing)}"
        )


def load_datasets(
    dataset_dir: Path,
    model_name: str,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    image_size = (128, 128) if model_name == "cnn" else (224, 224)
    color_mode = "grayscale" if model_name == "cnn" else "rgb"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=seed,
    )

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def build_cnn(num_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(128, 128, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def build_mobilenet(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def print_validation_report(model: tf.keras.Model, val_ds: tf.data.Dataset) -> None:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(predictions, axis=1))

    true_labels = np.concatenate(y_true)
    predicted_labels = np.concatenate(y_pred)

    print("Confusion matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    print()
    print(classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES))


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    validate_dataset(dataset_dir)

    train_ds, val_ds = load_datasets(
        dataset_dir=dataset_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.model == "cnn":
        model = build_cnn(len(CLASS_NAMES))
    else:
        model = build_mobilenet(len(CLASS_NAMES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print_validation_report(model, val_ds)


if __name__ == "__main__":
    main()
