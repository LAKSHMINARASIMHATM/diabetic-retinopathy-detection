"""
Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy
===============================================================================
Complete ML Pipeline: Data Preprocessing → Model Training → Evaluation → Inference

Requirements:
    pip install tensorflow opencv-python scikit-learn matplotlib seaborn numpy pandas albumentations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)
import albumentations as A
# NOTE: albumentations.tensorflow (ToTensorV2) was removed in modern albumentations.
# We do NOT need ToTensorV2 here because we handle numpy arrays directly.

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # Paths
    DATA_DIR        = "./data"
    TRAIN_DIR       = "./data/train"
    VAL_DIR         = "./data/val"
    TEST_DIR        = "./data/test"
    MODEL_SAVE_PATH = "./models/dr_model_best.keras"
    LOG_DIR         = "./logs"

    # Image settings
    IMG_SIZE        = 224          # EfficientNetB3 default
    CHANNELS        = 3
    INPUT_SHAPE     = (IMG_SIZE, IMG_SIZE, CHANNELS)

    # DR severity classes
    NUM_CLASSES     = 5
    CLASS_NAMES     = [
        "No DR",           # Grade 0
        "Mild DR",         # Grade 1
        "Moderate DR",     # Grade 2
        "Severe DR",       # Grade 3
        "Proliferative DR" # Grade 4
    ]
    CLASS_LABELS    = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Training hyperparameters
    BATCH_SIZE      = 32
    EPOCHS          = 50
    LEARNING_RATE   = 1e-4
    FINE_TUNE_LR    = 1e-5
    FINE_TUNE_EPOCH = 30          # epoch at which to unfreeze base model
    DROPOUT_RATE    = 0.4
    L2_REG          = 1e-4

    # Augmentation
    ROTATION_RANGE  = 30
    ZOOM_RANGE      = 0.2
    BRIGHTNESS_RANGE= (0.8, 1.2)

    SEED = 42


cfg = Config()
tf.random.set_seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ─────────────────────────────────────────────
# 2. PREPROCESSING UTILITIES
# ─────────────────────────────────────────────
class FundusPreprocessor:
    """
    Applies clinically motivated preprocessing steps to fundus images:
      - Green-channel extraction (highest contrast for retinal structures)
      - CLAHE (Contrast Limited Adaptive Histogram Equalization)
      - Circular crop to remove black borders
      - Resize & normalize
    """

    def __init__(self, img_size: int = cfg.IMG_SIZE):
        self.img_size = img_size

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel in LAB space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def circular_crop(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders with circular mask."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            image = image[y:y+h, x:x+w]
        return image

    def preprocess(self, image_path: str) -> np.ndarray:
        """Full preprocessing pipeline for a single fundus image."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img = self.circular_crop(img)
        img = self.apply_clahe(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0   # [0, 1]
        return img

    def preprocess_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """Preprocess an image already loaded as a numpy BGR array."""
        img = self.circular_crop(img_bgr)
        img = self.apply_clahe(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def batch_preprocess(self, image_paths: list) -> np.ndarray:
        images = []
        for p in image_paths:
            try:
                images.append(self.preprocess(p))
            except Exception as e:
                print(f"[WARNING] Skipping {p}: {e}")
        return np.array(images)


# ─────────────────────────────────────────────
# 3. DATA AUGMENTATION PIPELINE
# ─────────────────────────────────────────────
def get_augmentation_pipeline(is_training: bool = True) -> A.Compose:
    """
    Training augmentation uses retina-aware transforms.
    Returns values as numpy float32 arrays (NOT tensors).
    """
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=cfg.ROTATION_RANGE, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20,
                val_shift_limit=10, p=0.4),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])


# ─────────────────────────────────────────────
# 4. DATA PIPELINE (tf.data)
# ─────────────────────────────────────────────
class DRDataset:
    """
    Builds tf.data pipelines from a CSV file with columns:
        image_path, label  (0–4 DR grade)
    """

    def __init__(self, csv_path: str, preprocessor: FundusPreprocessor,
                 is_training: bool = True):
        self.df = pd.read_csv(csv_path)
        self.preprocessor = preprocessor
        self.augmentor = get_augmentation_pipeline(is_training)
        self.is_training = is_training

    def _load_and_augment(self, path, label):
        def _py_fn(p, lbl):
            p = p.numpy().decode("utf-8")
            img = self.preprocessor.preprocess(p)
            img_uint8 = (img * 255).astype(np.uint8)
            augmented = self.augmentor(image=img_uint8)
            img_out = augmented["image"].astype(np.float32)
            return img_out, lbl.numpy()

        img, lbl = tf.py_function(
            _py_fn, [path, label], [tf.float32, tf.int32])
        img.set_shape([cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS])
        lbl.set_shape([])
        return img, lbl

    def build(self, batch_size: int = cfg.BATCH_SIZE) -> tf.data.Dataset:
        paths  = self.df["image_path"].values
        labels = self.df["label"].values.astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if self.is_training:
            ds = ds.shuffle(buffer_size=len(paths), seed=cfg.SEED)
        ds = ds.map(self._load_and_augment,
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


# ─────────────────────────────────────────────
# 5. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class DRModel:
    """
    Transfer-learning model built on EfficientNetB3.

    Architecture:
        EfficientNetB3 (frozen) → GlobalAvgPool → BN → Dense(512) →
        Dropout → Dense(256) → Dropout → Dense(5, softmax)

    Two-stage training:
        Stage 1 – Train head only (base frozen)
        Stage 2 – Fine-tune full network at lower LR
    """

    def __init__(self):
        self.model = self._build()

    def _build(self) -> keras.Model:
        inputs = keras.Input(shape=cfg.INPUT_SHAPE, name="fundus_input")

        # ── Base Model ──────────────────────────────────────────────
        base = EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
        base.trainable = False   # freeze for stage-1

        # ── Custom Head ─────────────────────────────────────────────
        x = base.output
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="bn_head")(x)

        x = layers.Dense(
            512, activation="relu",
            kernel_regularizer=regularizers.l2(cfg.L2_REG),
            name="dense_512")(x)
        x = layers.Dropout(cfg.DROPOUT_RATE, name="drop_1")(x)

        x = layers.Dense(
            256, activation="relu",
            kernel_regularizer=regularizers.l2(cfg.L2_REG),
            name="dense_256")(x)
        x = layers.Dropout(cfg.DROPOUT_RATE / 2, name="drop_2")(x)

        outputs = layers.Dense(
            cfg.NUM_CLASSES, activation="softmax",
            name="dr_grade_output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="DR_EfficientNetB3")
        return model

    def compile(self, lr: float = cfg.LEARNING_RATE):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
            ]
        )

    def unfreeze_for_fine_tuning(self, num_layers_to_unfreeze: int = 30):
        """Unfreeze the top N layers of EfficientNetB3 for fine-tuning."""
        base = self.model.get_layer("efficientnetb3")
        base.trainable = True
        for layer in base.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        print(f"[Fine-tune] Unfrozen last {num_layers_to_unfreeze} base layers.")

    def summary(self):
        self.model.summary()


# ─────────────────────────────────────────────
# 6. TRAINING CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(stage: str = "stage1") -> list:
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH), exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=cfg.MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(cfg.LOG_DIR, stage),
            histogram_freq=1
        ),
    ]


# ─────────────────────────────────────────────
# 7. CLASS-WEIGHT CALCULATION
# ─────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray) -> dict:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    return {i: w for i, w in enumerate(weights)}


# ─────────────────────────────────────────────
# 8. TRAINING PIPELINE
# ─────────────────────────────────────────────
def train(train_csv: str, val_csv: str):
    preprocessor = FundusPreprocessor()

    # Datasets
    train_ds = DRDataset(train_csv, preprocessor, is_training=True).build()
    val_ds   = DRDataset(val_csv,   preprocessor, is_training=False).build()

    # Class weights (DR data is heavily imbalanced)
    train_labels = pd.read_csv(train_csv)["label"].values
    class_weights = compute_class_weights(train_labels)
    print(f"[Class Weights] {class_weights}")

    # Build & compile model
    dr_model = DRModel()
    dr_model.compile(lr=cfg.LEARNING_RATE)
    dr_model.summary()

    # ── Stage 1: Train head ──────────────────────────────────────
    print("\n=== STAGE 1: Training head (base frozen) ===")
    history1 = dr_model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.FINE_TUNE_EPOCH,
        class_weight=class_weights,
        callbacks=get_callbacks("stage1"),
    )

    # ── Stage 2: Fine-tune full network ─────────────────────────
    print("\n=== STAGE 2: Fine-tuning (base unfrozen) ===")
    dr_model.unfreeze_for_fine_tuning(num_layers_to_unfreeze=30)
    dr_model.compile(lr=cfg.FINE_TUNE_LR)

    history2 = dr_model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.EPOCHS - cfg.FINE_TUNE_EPOCH,
        class_weight=class_weights,
        callbacks=get_callbacks("stage2"),
    )

    plot_training_history(history1, history2)
    return dr_model, history1, history2


# ─────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model: keras.Model, test_csv: str):
    preprocessor = FundusPreprocessor()
    test_ds = DRDataset(test_csv, preprocessor, is_training=False).build(batch_size=1)

    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.append(preds[0])
        y_true.append(labels.numpy()[0])

    y_true      = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred,
                                 target_names=cfg.CLASS_NAMES))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # ROC-AUC (one-vs-rest)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(cfg.NUM_CLASSES)))
    auc = roc_auc_score(y_bin, y_pred_prob, multi_class="ovr", average="macro")
    print(f"\nMacro ROC-AUC: {auc:.4f}")

    plot_roc_curves(y_bin, y_pred_prob)
    return y_true, y_pred, y_pred_prob


# ─────────────────────────────────────────────
# 10. INFERENCE / PREDICTION
# ─────────────────────────────────────────────
class DRPredictor:
    """
    Loads a saved model and runs inference on new fundus images.

    Usage:
        predictor = DRPredictor("./models/dr_model_best.keras")
        result = predictor.predict("path/to/fundus.jpg")
        print(result)
    """

    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)
        self.preprocessor = FundusPreprocessor()
        self.augmentor = get_augmentation_pipeline(is_training=False)

    def predict(self, image_path: str, tta_rounds: int = 5) -> dict:
        """
        Predict DR grade with optional Test-Time Augmentation (TTA).
        """
        img = self.preprocessor.preprocess(image_path)
        return self._predict_from_array(img, tta_rounds)

    def predict_from_bytes(self, image_bytes: bytes, tta_rounds: int = 1) -> dict:
        """Predict directly from raw image bytes (for API use)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image bytes")
        img = self.preprocessor.preprocess_array(img_bgr)
        return self._predict_from_array(img, tta_rounds)

    def _predict_from_array(self, img: np.ndarray, tta_rounds: int = 1) -> dict:
        img_uint8 = (img * 255).astype(np.uint8)
        tta_aug = get_augmentation_pipeline(is_training=True) if tta_rounds > 1 \
            else self.augmentor

        preds = []
        for _ in range(tta_rounds):
            aug = tta_aug(image=img_uint8)
            x = aug["image"].astype(np.float32)
            x = np.expand_dims(x, axis=0)
            preds.append(self.model.predict(x, verbose=0)[0])

        probs = np.mean(preds, axis=0)
        grade = int(np.argmax(probs))
        confidence = float(probs[grade])

        return {
            "grade":          grade,
            "class_name":     cfg.CLASS_NAMES[grade],
            "confidence":     round(confidence * 100, 2),
            "probabilities":  {cfg.CLASS_NAMES[i]: round(float(p)*100, 2)
                               for i, p in enumerate(probs)},
            "risk_level":     self._risk_level(grade),
            "recommendation": self._recommendation(grade),
        }

    @staticmethod
    def _risk_level(grade: int) -> str:
        return ["Low", "Low-Moderate", "Moderate", "High", "Critical"][grade]

    @staticmethod
    def _recommendation(grade: int) -> str:
        recs = [
            "Routine annual screening.",
            "Follow-up in 9–12 months. Optimize glycemic control.",
            "Ophthalmology referral within 3–6 months.",
            "Urgent ophthalmology referral within 1–3 months.",
            "Immediate ophthalmology referral. Laser therapy may be required.",
        ]
        return recs[grade]


# ─────────────────────────────────────────────
# 11. GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────
def grad_cam(model: keras.Model, image: np.ndarray,
             last_conv_layer: str = "top_activation") -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model interpretability.
    Highlights retinal regions that influenced the prediction.
    """
    # In Keras 3 / TF 2.20 the EfficientNetB3 layers are flattened into
    # the top-level model, so get_layer should work directly.
    # Fallback: try finding a suitable conv/activation layer automatically.
    target_layer = None
    try:
        target_layer = model.get_layer(last_conv_layer)
    except ValueError:
        # Find the last activation or conv layer before the GAP
        for layer in reversed(model.layers):
            if "conv" in layer.name or "activation" in layer.name:
                target_layer = layer
                break
        if target_layer is None:
            print("[Grad-CAM] Could not find a suitable conv layer. Skipping.")
            return (image * 255).astype(np.uint8)

    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    img_tensor = tf.cast(np.expand_dims(image, 0), tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        pred_class = tf.argmax(preds[0])
        class_score = preds[:, pred_class]

    grads  = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam    = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    cam    = tf.nn.relu(cam).numpy()

    # Normalize and resize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay on original image
    original = (image * 255).astype(np.uint8)
    superimposed = cv2.addWeighted(
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# 12. PLOTTING UTILITIES
# ─────────────────────────────────────────────
def plot_training_history(h1, h2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(
            axes,
            [("accuracy", "val_accuracy"), ("loss", "val_loss")],
            ["Accuracy", "Loss"]):
        s1 = getattr(h1, "history", h1)[metric[0]]
        s2 = getattr(h2, "history", h2)[metric[0]]
        v1 = getattr(h1, "history", h1)[metric[1]]
        v2 = getattr(h2, "history", h2)[metric[1]]
        combined_train = s1 + s2
        combined_val   = v1 + v2
        ax.plot(combined_train, label="Train", color="#2196F3")
        ax.plot(combined_val,   label="Val",   color="#FF5722")
        ax.axvline(len(s1), color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(f"Training {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./training_history.png", dpi=150)
    plt.show()
    print("[Saved] training_history.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASS_NAMES,
                yticklabels=cfg.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Diabetic Retinopathy Grading",
                 fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=150)
    plt.show()
    print("[Saved] confusion_matrix.png")


def plot_roc_curves(y_bin, y_pred_prob):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    for i, (cls, color) in enumerate(zip(cfg.CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_pred_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Per DR Grade", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./roc_curves.png", dpi=150)
    plt.show()
    print("[Saved] roc_curves.png")


# ─────────────────────────────────────────────
# 13. DEMO (SYNTHETIC DATA)
# ─────────────────────────────────────────────
def create_synthetic_demo():
    """
    Creates a tiny synthetic dataset of random images for pipeline demo.
    Replace with real fundus data (e.g. APTOS 2019, EyePACS, IDRiD).
    """
    print("[Demo] Creating synthetic fundus-like images …")
    for split in ["train", "val", "test"]:
        os.makedirs(f"./data/{split}/images", exist_ok=True)

    records = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(42)

    for split, n in [("train", 100), ("val", 20), ("test", 20)]:
        for i in range(n):
            label = rng.integers(0, 5)
            fname = f"./data/{split}/images/{split}_{i:04d}.png"
            # Simulate retinal image (circular green blob on black)
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.circle(img, (112, 112), 100,
                       (int(rng.integers(30,80)),
                        int(rng.integers(80,180)),
                        int(rng.integers(20,70))), -1)
            # Add synthetic lesions for non-zero grades
            for _ in range(label * 5):
                cx, cy = rng.integers(30,194), rng.integers(30,194)
                cv2.circle(img, (cx,cy), rng.integers(3,8),
                           (200,50,50), -1)
            cv2.imwrite(fname, img)
            records[split].append({"image_path": fname, "label": int(label)})

    for split, rows in records.items():
        pd.DataFrame(rows).to_csv(f"./data/{split}.csv", index=False)

    print("[Demo] Synthetic dataset created.")


# ─────────────────────────────────────────────
# 14. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── 1. Prepare data ────────────────────────────────────────────
    create_synthetic_demo()   # ← replace with your real data path

    # ── 2. Train ───────────────────────────────────────────────────
    dr_model, h1, h2 = train(
        train_csv="./data/train.csv",
        val_csv="./data/val.csv"
    )

    # ── 3. Evaluate ────────────────────────────────────────────────
    evaluate(dr_model.model, test_csv="./data/test.csv")

    # ── 4. Single-image inference ──────────────────────────────────
    predictor = DRPredictor(cfg.MODEL_SAVE_PATH)
    sample_image = "./data/test/images/test_0000.png"
    result = predictor.predict(sample_image, tta_rounds=5)

    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    for k, v in result.items():
        print(f"  {k:>15}: {v}")

    # ── 5. Grad-CAM ────────────────────────────────────────────────
    preprocessor = FundusPreprocessor()
    img = preprocessor.preprocess(sample_image)
    heatmap = grad_cam(dr_model.model, img, last_conv_layer="top_activation")
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap); plt.axis("off")
    plt.title("Grad-CAM – Retinal Attention Map")
    plt.tight_layout()
    plt.savefig("./grad_cam.png", dpi=150)
    plt.show()
    print("[Saved] grad_cam.png")
