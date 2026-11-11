
import os
import csv
import random

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

DAY6_ACCURACY = 0.8253

DATA_DIR = "../data/fashion-product-images-small"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------

print("Loading Fashion Product Images dataset...")

image_paths = []
labels = []

with open(CSV_PATH, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        image_id = row["id"]
        category = row["articleType"]

        image_path = os.path.join(
            IMAGE_DIR, f"{image_id}.jpg"
        )

        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(category)


# --------------------------------------------------
# STEP 2: PREPARE CLASS LABELS
# --------------------------------------------------

class_counts = {}

for label in labels:
    class_counts[label] = class_counts.get(label, 0) + 1

# Keep categories with at least 3 images
valid_classes = sorted(
    label for label, count in class_counts.items()
    if count >= 3
)

class_to_index = {
    name: index
    for index, name in enumerate(valid_classes)
}

print("Number of categories:", len(valid_classes))

class_names = valid_classes

# Group image paths by category
class_images = {
    name: []
    for name in class_names
}

for path, label in zip(image_paths, labels):
    if label in class_to_index:
        class_images[label].append(path)


# --------------------------------------------------
# STEP 3: TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

train_paths = []
train_labels = []

val_paths = []
val_labels = []

test_paths = []
test_labels = []

for category in class_names:

    images = class_images[category]

    random.shuffle(images)

    total = len(images)

    test_count = max(1, int(total * 0.10))
    val_count = max(1, int(total * 0.10))

    train_images = images[:total - test_count - val_count]
    val_images = images[total - test_count - val_count:
                        total - test_count]
    test_images = images[total - test_count:]

    class_index = class_to_index[category]

    train_paths.extend(train_images)
    train_labels.extend([class_index] * len(train_images))

    val_paths.extend(val_images)
    val_labels.extend([class_index] * len(val_images))

    test_paths.extend(test_images)
    test_labels.extend([class_index] * len(test_images))


print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))


# --------------------------------------------------
# STEP 4: IMAGE PREPROCESSING
# --------------------------------------------------

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32)

    return image, label


def create_dataset(paths, targets, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, targets)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_dataset(
    val_paths,
    val_labels
)

test_ds = create_dataset(
    test_paths,
    test_labels
)


# --------------------------------------------------
# STEP 5: LOAD PRETRAINED VGG16
# --------------------------------------------------

print("\nLoading pretrained VGG16...")

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all pretrained layers
base_model.trainable = False


# --------------------------------------------------
# STEP 6: BUILD CLASSIFICATION HEAD
# --------------------------------------------------

inputs = layers.Input(
    shape=(224, 224, 3)
)

# Data augmentation
x = layers.RandomFlip("horizontal")(inputs)

x = layers.RandomRotation(0.05)(x)

# VGG16-specific preprocessing
x = layers.Lambda(preprocess_input)(x)

# Pretrained feature extractor
x = base_model(x, training=False)

# New classification head
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# --------------------------------------------------
# STEP 7: TRAIN CLASSIFICATION HEAD
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining classification head...")

head_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

head_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=[head_early_stopping]
)


# --------------------------------------------------
# STEP 8: FINE-TUNE TOP VGG16 LAYERS
# --------------------------------------------------

print("\nFine-tuning VGG16 top layers...")

base_model.trainable = True

# Freeze all layers except the final convolution block
for layer in base_model.layers:

    if layer.name.startswith("block5_"):
        layer.trainable = True
    else:
        layer.trainable = False


# Recompile with a much smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[fine_tune_early_stopping]
)


# --------------------------------------------------
# STEP 9: EVALUATE MODEL
# --------------------------------------------------

print("\nEvaluating Day 7 VGG16 model...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nDAY 7 RESULTS")
print("-------------------------")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Accuracy (%):", round(test_accuracy * 100, 2))


# --------------------------------------------------
# STEP 10: COMPARE DAY 6 AND DAY 7
# --------------------------------------------------

comparison_path = os.path.join(
    MODEL_DIR,
    "day7_model_comparison.csv"
)

with open(
    comparison_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Test Accuracy",
        "Test Loss"
    ])

    writer.writerow([
        "Day 6 CNN",
        DAY6_ACCURACY,
        ""
    ])

    writer.writerow([
        "Day 7 VGG16 Transfer Learning",
        round(float(test_accuracy), 4),
        round(float(test_loss), 4)
    ])


# --------------------------------------------------
# STEP 11: SAVE MODEL AND CLASS NAMES
# --------------------------------------------------

model.save(
    os.path.join(
        MODEL_DIR,
        "retailmind_vgg16.keras"
    )
)

with open(
    os.path.join(
        MODEL_DIR,
        "retailmind_vgg16_classes.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(class_names))


# --------------------------------------------------
# STEP 12: SAVE TRAINING HISTORY
# --------------------------------------------------

history_path = os.path.join(
    MODEL_DIR,
    "day7_training_history.csv"
)

with open(
    history_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Phase",
        "Epoch",
        "Accuracy",
        "Validation Accuracy",
        "Loss",
        "Validation Loss"
    ])

    for phase, history in [
        ("Classification Head", head_history),
        ("Fine Tuning", fine_tune_history)
    ]:

        for index in range(len(history.history["accuracy"])):

            writer.writerow([
                phase,
                index + 1,
                history.history["accuracy"][index],
                history.history["val_accuracy"][index],
                history.history["loss"][index],
                history.history["val_loss"][index]
            ])


print("\nDay 7 completed successfully!")

print("Model saved: models/retailmind_vgg16.keras")
print("Classes saved: models/retailmind_vgg16_classes.txt")
print("Comparison saved: models/day7_model_comparison.csv")
print("History saved: models/day7_training_history.csv")
