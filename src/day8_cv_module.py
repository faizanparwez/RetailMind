
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "retailmind_cnn.keras"
MODELS_DIR = ROOT_DIR / "models"

OUTPUT_DIR = ROOT_DIR / "outputs" / "day8"
SEGMENTED_DIR = OUTPUT_DIR / "segmented"
HEATMAP_DIR = OUTPUT_DIR / "gradcam"

IMAGE_SIZE = (128, 128)


# --------------------------------------------------
# RETAILMIND COMPUTER VISION MODULE
# --------------------------------------------------

class RetailMindCV:

    def __init__(self, model_path=MODEL_PATH):

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Classification model not found: {model_path}"
            )

        self.model = tf.keras.models.load_model(model_path)

        # Build the model's computation graph
        dummy_input = tf.zeros((1, 128, 128, 3))
        self.model(dummy_input)

        self.class_names = self._load_class_names()

        self.last_conv_layer = self._find_last_conv_layer()

        self.gradcam_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.last_conv_layer.output,
                self.model.output
            ]
        )

        SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # LOAD CLASS LABELS
    # --------------------------------------------------

    def _load_class_names(self):

        candidates = [
            MODELS_DIR / "retailmind_classes.txt",
            MODELS_DIR / "retailmind_cnn_classes.txt",
            MODELS_DIR / "classes.txt"
        ]

        for path in candidates:

            if path.exists():
                return [
                    line.strip()
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]

        # Search for another saved class-name file.
        for path in MODELS_DIR.glob("*class*.txt"):

            names = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip()
            ]

            if names:
                return names

        return None

    # --------------------------------------------------
    # FIND LAST CONVOLUTIONAL LAYER
    # --------------------------------------------------

    def _find_last_conv_layer(self):

        for layer in reversed(self.model.layers):

            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer

        raise ValueError(
            "No Conv2D layer found. Grad-CAM requires "
            "a convolutional classification model."
        )

    # --------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------

    def preprocess_image(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, IMAGE_SIZE)

        normalized = resized.astype("float32") / 255.0

        batch = np.expand_dims(normalized, axis=0)

        return image, batch

    # --------------------------------------------------
    # PRODUCT CLASSIFICATION
    # --------------------------------------------------

    def classify(self, image_path):

        original, batch = self.preprocess_image(image_path)

        predictions = self.model.predict(batch, verbose=0)[0]

        class_id = int(np.argmax(predictions))

        confidence = float(predictions[class_id])

        if self.class_names and class_id < len(self.class_names):
            label = self.class_names[class_id]
        else:
            label = f"Class {class_id}"

        return {
            "label": label,
            "class_id": class_id,
            "confidence": confidence
        }

    # --------------------------------------------------
    # PRODUCT SEGMENTATION USING OPENCV GRABCUT
    # --------------------------------------------------

    def segment_product(self, image):

        height, width = image.shape[:2]

        mask = np.zeros((height, width), np.uint8)

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)

        rectangle = (
            int(width * 0.05),
            int(height * 0.05),
            int(width * 0.90),
            int(height * 0.90)
        )

        try:

            cv2.grabCut(
                image,
                mask,
                rectangle,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_RECT
            )

            binary_mask = np.where(
                (mask == cv2.GC_FGD) |
                (mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype("uint8")

        except cv2.error:

            # Fallback if GrabCut cannot initialize.
            binary_mask = np.zeros(
                (height, width),
                dtype="uint8"
            )

            x, y, w, h = rectangle

            binary_mask[y:y+h, x:x+w] = 255

        return binary_mask

    # --------------------------------------------------
    # REMOVE BACKGROUND AND COMPOSITE ON WHITE
    # --------------------------------------------------

    def remove_background(self, image, mask):

        white_background = np.full_like(image, 255)

        foreground = cv2.bitwise_and(
            image,
            image,
            mask=mask
        )

        background = cv2.bitwise_and(
            white_background,
            white_background,
            mask=cv2.bitwise_not(mask)
        )

        result = cv2.add(foreground, background)

        return result

    # --------------------------------------------------
    # PRODUCT BOUNDING BOX FROM SEGMENTATION
    # --------------------------------------------------

    def detect_product(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, width, height = cv2.boundingRect(
            largest_contour
        )

        return {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height)
        }

    # --------------------------------------------------
    # GRAD-CAM EXPLAINABILITY
    # --------------------------------------------------

    def generate_gradcam(self, image_path):

        original, batch = self.preprocess_image(image_path)

        input_tensor = tf.convert_to_tensor(batch)

        with tf.GradientTape() as tape:

            conv_outputs, predictions = self.gradcam_model(
                input_tensor,
                training=False
            )

            class_id = tf.argmax(
                predictions[0]
            )

            class_score = predictions[:, class_id]

        gradients = tape.gradient(
            class_score,
            conv_outputs
        )

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_gradients,
            axis=-1
        )

        heatmap = tf.maximum(heatmap, 0)

        heatmap = heatmap / (
            tf.reduce_max(heatmap) + 1e-8
        )

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (original.shape[1], original.shape[0])
        )

        heatmap = np.uint8(255 * heatmap)

        colored_heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original,
            0.6,
            colored_heatmap,
            0.4,
            0
        )

        return overlay

    # --------------------------------------------------
    # COMPLETE END-TO-END PIPELINE
    # --------------------------------------------------

    def process_image(self, image_path):

        image_path = Path(image_path)

        original, _ = self.preprocess_image(image_path)

        # Classification
        classification = self.classify(image_path)

        # Segmentation
        mask = self.segment_product(original)

        # Background removal
        clean_image = self.remove_background(
            original,
            mask
        )

        # Bounding box
        bounding_box = self.detect_product(mask)

        # Grad-CAM
        heatmap = self.generate_gradcam(image_path)

        # Save outputs
        image_name = image_path.stem

        segmented_path = SEGMENTED_DIR / f"{image_name}_clean.jpg"
        mask_path = SEGMENTED_DIR / f"{image_name}_mask.png"
        heatmap_path = HEATMAP_DIR / f"{image_name}_gradcam.jpg"

        cv2.imwrite(str(segmented_path), clean_image)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(heatmap_path), heatmap)

        return {
            "classification": classification,
            "bounding_box": bounding_box,
            "segmented_image": str(segmented_path),
            "mask": str(mask_path),
            "gradcam": str(heatmap_path)
        }


# --------------------------------------------------
# COMMAND LINE INTERFACE
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RetailMind AI Day 8 Computer Vision"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the product image"
    )

    args = parser.parse_args()

    cv_module = RetailMindCV()

    result = cv_module.process_image(args.image)

    print("\nRETAILMIND CV RESULTS")
    print("---------------------")

    print("Classification:", result["classification"])
    print("Bounding Box:", result["bounding_box"])

    print("Clean Image:", result["segmented_image"])
    print("Segmentation Mask:", result["mask"])
    print("Grad-CAM:", result["gradcam"])
