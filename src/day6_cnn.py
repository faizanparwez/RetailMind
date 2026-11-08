import csv
import random
from pathlib import Path
from collections import defaultdict

import tensorflow as tf
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

IMAGE_DIR = Path("../data/fashion-product-images-small/images")
CSV_PATH = Path("../data/fashion-product-images-small/styles.csv")
MODEL_DIR = Path("../models")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------------------------------------------------
# 2. LOAD CSV AND MATCH IMAGES
# --------------------------------------------------

print("Loading product dataset...")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find: {CSV_PATH}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Could not find: {IMAGE_DIR}")

records_by_class = defaultdict(list)

with open(CSV_PATH, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required_columns = {"id", "articleType"}

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            "styles.csv must contain 'id' and 'articleType' columns."
        )

    for row in reader:
        product_id = (row.get("id") or "").strip()
        category = (row.get("articleType") or "").strip()

        if not product_id or not category:
            continue

        image_path = IMAGE_DIR / f"{product_id}.jpg"

        if image_path.exists():
            records_by_class[category].append(str(image_path))


# Remove categories with fewer than 3 images.
# At least one image is needed for each of train, validation and test.
records_by_class = {
    category: paths
    for category, paths in records_by_class.items()
    if len(paths) >= 3
}

class_names = sorted(records_by_class.keys())
class_to_index = {
    name: index for index, name in enumerate(class_names)
}

if len(class_names) < 2:
    raise ValueError(
        "Not enough valid categories found. Check the dataset folder and CSV."
    )

print(f"Number of categories: {len(class_names)}")
print(f"Number of usable images: {sum(len(v) for v in records_by_class.values())}")


# --------------------------------------------------
# 3. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

train_paths = []
train_labels = []

val_paths = []
val_labels = []

test_paths = []
test_labels = []

for category, paths in records_by_class.items():

    random.shuffle(paths)

    total = len(paths)

    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_items = paths[:train_end]
    val_items = paths[train_end:val_end]
    test_items = paths[val_end:]

    label = class_to_index[category]

    train_paths.extend(train_items)
    train_labels.extend([label] * len(train_items))

    val_paths.extend(val_items)
    val_labels.extend([label] * len(val_items))

    test_paths.extend(test_items)
    test_labels.extend([label] * len(test_items))


print("\nDataset split:")
print(f"Training images:   {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Testing images:    {len(test_paths)}")


# --------------------------------------------------
# 4. IMAGE LOADING PIPELINE
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

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
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
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(
    train_paths, train_labels, training=True
)

val_ds = create_dataset(
    val_paths, val_labels
)

test_ds = create_dataset(
    test_paths, test_labels
)


# --------------------------------------------------
# 5. DATA AUGMENTATION
# --------------------------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1)
], name="data_augmentation")


# --------------------------------------------------
# 6. BUILD CNN MODEL
# --------------------------------------------------

num_classes = len(class_names)

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    ),

    augmentation,

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        num_classes, activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "retailmind_cnn_best.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]


print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# 9. GENERATE CONFUSION MATRIX
# --------------------------------------------------

print("\nGenerating confusion matrix...")

predictions = model.predict(test_ds)

predicted_labels = tf.argmax(
    predictions,
    axis=1
)

actual_labels = tf.concat(
    [labels for images, labels in test_ds],
    axis=0
)

confusion_matrix = tf.math.confusion_matrix(
    actual_labels,
    predicted_labels,
    num_classes=num_classes
).numpy()


# Save confusion matrix as CSV
confusion_path = MODEL_DIR / "day6_confusion_matrix.csv"

with open(confusion_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow(["Actual / Predicted"] + class_names)

    for index, row in enumerate(confusion_matrix):
        writer.writerow([class_names[index]] + row.tolist())


# --------------------------------------------------
# 10. PLOT TRAINING HISTORY
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("RetailMind CNN Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "day6_cnn_accuracy.png"
)

plt.show()


# --------------------------------------------------
# 11. SAVE MODEL AND CLASS LABELS
# --------------------------------------------------

model.save(
    MODEL_DIR / "retailmind_cnn.keras"
)

with open(
    MODEL_DIR / "retailmind_cnn_classes.txt",
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:
        file.write(class_name + "\n")


print("\nDay 6 completed successfully!")
print("Model saved: models/retailmind_cnn.keras")
print("Classes saved: models/retailmind_cnn_classes.txt")
print("Confusion matrix: models/day6_confusion_matrix.csv")
print("Accuracy graph: models/day6_cnn_accuracy.png")