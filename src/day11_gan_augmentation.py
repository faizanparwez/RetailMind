
"""
RetailMind AI — Day 11
GAN-Based Synthetic Fashion Image Augmentation

Libraries: TensorFlow / Keras
Dataset: Fashion Product Images (Kaggle)
"""

import os
import csv
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "fashion-product-images-small"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = DATA_DIR / "synthetic_images"

IMAGE_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100

BATCH_SIZE = 32
EPOCHS = 100
GENERATE_COUNT = 500

SEED = 42

tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD DATASET METADATA
# ============================================================

def load_metadata():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {CSV_PATH}"
        )

    records = []

    with open(CSV_PATH, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("styles.csv is empty.")

        print("Available columns:", reader.fieldnames)

        category_column = (
            "articleType"
            if "articleType" in reader.fieldnames
            else "masterCategory"
        )

        if category_column not in reader.fieldnames:
            raise ValueError(
                "No articleType or masterCategory column found."
            )

        for row in reader:

            category = row.get(category_column, "").strip()
            product_id = row.get("id", "").strip()

            if not category or not product_id:
                continue

            image_path = IMAGE_DIR / f"{product_id}.jpg"

            if image_path.exists():
                records.append({
                    "id": product_id,
                    "category": category,
                    "path": str(image_path)
                })

    if not records:
        raise ValueError("No matching product images found.")

    print(f"\nTotal usable images: {len(records)}")

    return records


# ============================================================
# 3. IDENTIFY UNDER-REPRESENTED CATEGORIES
# ============================================================

def analyze_categories(records):

    category_counts = {}

    for record in records:

        category = record["category"]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    print("\nCATEGORY DISTRIBUTION")
    print("-" * 40)

    for category, count in sorted(
        category_counts.items(),
        key=lambda item: item[1]
    ):

        print(f"{category:<25} {count}")

    return category_counts


# ============================================================
# 4. LOAD AND PREPROCESS CATEGORY IMAGES
# ============================================================

def preprocess_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_jpeg(
        image,
        channels=CHANNELS
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    # Normalize pixel values from [0, 255] to [-1, 1]
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    return image


def create_dataset(records, selected_category):

    selected_records = [
        record for record in records
        if record["category"] == selected_category
    ]

    if len(selected_records) < 100:
        raise ValueError(
            "Selected category has fewer than 100 images. "
            "Choose a category with more training examples."
        )

    print(
        f"\nTraining GAN on {selected_category}: "
        f"{len(selected_records)} images"
    )

    paths = [record["path"] for record in selected_records]

    dataset = tf.data.Dataset.from_tensor_slices(paths)

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.shuffle(
        buffer_size=min(len(paths), 2000),
        seed=SEED
    )

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(selected_records)


# ============================================================
# 5. BUILD GENERATOR
# ============================================================

def build_generator():

    model = keras.Sequential(
        [
            layers.Input(shape=(LATENT_DIM,)),

            layers.Dense(8 * 8 * 256, use_bias=False),

            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Reshape((8, 8, 256)),

            layers.Conv2DTranspose(
                128, 4,
                strides=2,
                padding="same",
                use_bias=False
            ),

            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2DTranspose(
                64, 4,
                strides=2,
                padding="same",
                use_bias=False
            ),

            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2DTranspose(
                CHANNELS, 4,
                strides=2,
                padding="same",
                activation="tanh"
            )
        ],
        name="Generator"
    )

    return model


# ============================================================
# 6. BUILD DISCRIMINATOR
# ============================================================

def build_discriminator():

    model = keras.Sequential(
        [
            layers.Input(
                shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
            ),

            layers.Conv2D(
                64, 4,
                strides=2,
                padding="same"
            ),

            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),

            layers.Conv2D(
                128, 4,
                strides=2,
                padding="same"
            ),

            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),

            layers.Flatten(),

            layers.Dense(1)
        ],
        name="Discriminator"
    )

    return model


# ============================================================
# 7. GAN TRAINING CLASS
# ============================================================

class RetailMindGAN(keras.Model):

    def __init__(self, generator, discriminator):

        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.g_loss_tracker = keras.metrics.Mean(
            name="generator_loss"
        )

        self.d_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss"
        )

    @property
    def metrics(self):

        return [
            self.g_loss_tracker,
            self.d_loss_tracker
        ]

    def compile(self, g_optimizer, d_optimizer):

        super().compile()

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=True
        )

    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]

        noise = tf.random.normal(
            [batch_size, LATENT_DIM]
        )

        # ----------------------------------------
        # TRAIN DISCRIMINATOR
        # ----------------------------------------

        with tf.GradientTape() as d_tape:

            fake_images = self.generator(
                noise,
                training=True
            )

            real_output = self.discriminator(
                real_images,
                training=True
            )

            fake_output = self.discriminator(
                tf.stop_gradient(fake_images),
                training=True
            )

            real_loss = self.loss_fn(
                tf.ones_like(real_output),
                real_output
            )

            fake_loss = self.loss_fn(
                tf.zeros_like(fake_output),
                fake_output
            )

            d_loss = real_loss + fake_loss

        d_gradients = d_tape.gradient(
            d_loss,
            self.discriminator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(
                d_gradients,
                self.discriminator.trainable_variables
            )
        )

        # ----------------------------------------
        # TRAIN GENERATOR
        # ----------------------------------------

        noise = tf.random.normal(
            [batch_size, LATENT_DIM]
        )

        with tf.GradientTape() as g_tape:

            generated_images = self.generator(
                noise,
                training=True
            )

            fake_output = self.discriminator(
                generated_images,
                training=True
            )

            g_loss = self.loss_fn(
                tf.ones_like(fake_output),
                fake_output
            )

        g_gradients = g_tape.gradient(
            g_loss,
            self.generator.trainable_variables
        )

        self.g_optimizer.apply_gradients(
            zip(
                g_gradients,
                self.generator.trainable_variables
            )
        )

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {
            "generator_loss": self.g_loss_tracker.result(),
            "discriminator_loss": self.d_loss_tracker.result()
        }


# ============================================================
# 8. SAVE GENERATED IMAGE SAMPLES
# ============================================================

def save_generated_images(generator, category, count):

    output_folder = OUTPUT_DIR / category

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_count = 0

    for start in range(0, count, BATCH_SIZE):

        current_batch = min(
            BATCH_SIZE,
            count - start
        )

        noise = tf.random.normal(
            [current_batch, LATENT_DIM]
        )

        generated_images = generator(
            noise,
            training=False
        )

        # Convert [-1, 1] back to [0, 255]
        generated_images = (
            (generated_images + 1.0) * 127.5
        )

        generated_images = tf.clip_by_value(
            generated_images,
            0,
            255
        )

        generated_images = tf.cast(
            generated_images,
            tf.uint8
        )

        for image in generated_images:

            image_bytes = tf.io.encode_png(image)

            image_path = (
                output_folder /
                f"synthetic_{saved_count:05d}.png"
            )

            tf.io.write_file(
                str(image_path),
                image_bytes
            )

            saved_count += 1

    print(
        f"\nSaved {saved_count} synthetic images "
        f"to {output_folder}"
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Product category to augment"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    parser.add_argument(
        "--generate",
        type=int,
        default=GENERATE_COUNT
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.generate < 1:
        raise ValueError(
            "Epochs and generation count must be positive."
        )

    print("=" * 55)
    print("RETAILMIND AI — DAY 11 GAN AUGMENTATION")
    print("=" * 55)

    MODEL_DIR.mkdir(exist_ok=True)

    records = load_metadata()

    category_counts = analyze_categories(records)

    # Automatically select the smallest category
    # unless a category is explicitly provided.

    if args.category:

        selected_category = args.category

        if selected_category not in category_counts:
            raise ValueError(
                f"Unknown category: {selected_category}"
            )

    else:

        selected_category = min(
            category_counts,
            key=category_counts.get
        )

    print(
        f"\nSelected category: {selected_category}"
    )

    dataset, real_count = create_dataset(
        records,
        selected_category
    )

    # Build models

    generator = build_generator()
    discriminator = build_discriminator()

    gan = RetailMindGAN(
        generator,
        discriminator
    )

    gan.compile(
        g_optimizer=keras.optimizers.Adam(
            learning_rate=0.0002,
            beta_1=0.5
        ),

        d_optimizer=keras.optimizers.Adam(
            learning_rate=0.0002,
            beta_1=0.5
        )
    )

    print("\nGenerator summary:")
    generator.summary()

    print("\nDiscriminator summary:")
    discriminator.summary()

    # Train GAN

    print("\nStarting GAN training...")

    gan.fit(
        dataset,
        epochs=args.epochs
    )

    # Save trained generator

    generator_path = (
        MODEL_DIR / "day11_generator.keras"
    )

    generator.save(generator_path)

    print(
        f"\nGenerator saved: {generator_path}"
    )

    # Generate synthetic images

    save_generated_images(
        generator,
        selected_category,
        args.generate
    )

    # Save run metadata

    metadata_path = (
        OUTPUT_DIR / "day11_run_metadata.csv"
    )

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "category",
            "real_training_images",
            "synthetic_images",
            "epochs",
            "image_size"
        ])

        writer.writerow([
            selected_category,
            real_count,
            args.generate,
            args.epochs,
            IMAGE_SIZE
        ])

    print("\nDAY 11 COMPLETED!")
    print("Inspect generated images before using them.")
    print("Do not automatically mix unverified images into training.")


if __name__ == "__main__":
    main()
