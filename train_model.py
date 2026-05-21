"""
Train an acne severity classification model using MobileNetV2 transfer learning.
Dataset structure expected:
    dataset/
        mild/
        moderate/
        severe/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODEL_PATH   = "dataset/model/acne_model.h5"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 20
SEED         = 42
CLASSES      = ["mild_new", "moderate_new", "severe_new"]

# ── Data generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    subset="training",
    seed=SEED,
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    subset="validation",
    seed=SEED,
)

print(f"\nClass indices : {train_gen.class_indices}")
print(f"Training   samples : {train_gen.samples}")
print(f"Validation samples : {val_gen.samples}\n")

# ── Build model (MobileNetV2 + custom head) ────────────────────────────────────
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # freeze base initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(len(CLASSES), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
]

# ── Phase 1: Train head only ──────────────────────────────────────────────────
print("\n=== Phase 1: Training classification head ===")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Phase 2: Fine-tune top layers of base ─────────────────────────────────────
print("\n=== Phase 2: Fine-tuning top 30 layers of MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n=== Evaluation on validation set ===")
val_gen.reset()
y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes

print(classification_report(y_true, y_pred, target_names=CLASSES))

# ── Plot training curves ───────────────────────────────────────────────────────
def merge_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

epochs_range = range(len(merge_histories(history1, history2, "accuracy")))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(
    axes,
    [("accuracy", "val_accuracy"), ("loss", "val_loss")],
    ["Accuracy", "Loss"],
):
    ax.plot(epochs_range, merge_histories(history1, history2, metric[0]), label="Train")
    ax.plot(epochs_range, merge_histories(history1, history2, metric[1]), label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("\nTraining curves saved to training_curves.png")
print(f"Model saved to {MODEL_PATH}")
