"""
Predict acne severity from a single image.

Usage:
    python predict.py path/to/image.jpg
"""

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "dataset/model/acne_model.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["Mild", "Moderate", "Severe"]

ADVICE = {
    "Mild": (
        "Your acne appears mild. Maintain a gentle cleansing routine, "
        "use non-comedogenic products, and stay hydrated."
    ),
    "Moderate": (
        "Moderate acne detected. Consider over-the-counter benzoyl peroxide "
        "or salicylic acid treatments, and consult a dermatologist if it persists."
    ),
    "Severe": (
        "Severe acne detected. Please consult a dermatologist as soon as possible "
        "for prescription-grade treatment options."
    ),
}


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_path: str):
    model = load_model()
    img   = preprocess(image_path)
    probs = model.predict(img)[0]

    pred_idx   = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    print(f"\n{'='*40}")
    print(f"  Image     : {image_path}")
    print(f"  Prediction: {pred_label}  ({confidence:.1f}% confidence)")
    print(f"{'='*40}")
    print("\nProbabilities:")
    for cls, prob in zip(CLASSES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<10} {prob*100:5.1f}%  {bar}")
    print(f"\nAdvice: {ADVICE[pred_label]}\n")

    return pred_label, confidence, probs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
