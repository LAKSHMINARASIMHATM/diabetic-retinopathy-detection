"""
FastAPI backend for Diabetic Retinopathy Detection
===================================================
Real-world dual-model inference system:
  - EfficientNetB3 (Keras)   → fundus image grading
  - Random Forest (joblib)   → clinical feature assessment

Endpoints:
  POST /predict          – Upload fundus image → DR grade + Grad-CAM
  POST /predict-clinical – Submit clinical form → DR grade
  GET  /health           – Health check
  GET  /info             – Model info & class names
  GET  /model-metadata   – Training metadata for the clinical model

Run:
    uvicorn server:app --reload --port 8000
"""

import os
import io
import sys
import time
import base64
import logging
import json

# ── Suppress GPU/CUDA warnings on CPU Space ────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import Optional

import numpy as np

# ── FastAPI / HTTP ──────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Add project root to path ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Dual-model DR grading: EfficientNetB3 (images) + Random Forest (clinical features)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model paths ──────────────────────────────────────────────
MODEL_PATH    = str(Path(__file__).parent / "models" / "dr_model_best.keras")
CLINICAL_PATH = str(Path(__file__).parent / "models" / "dr_clinical_model.joblib")
SCALER_PATH   = str(Path(__file__).parent / "models" / "dr_scaler.joblib")
META_PATH     = str(Path(__file__).parent / "models" / "dr_model_metadata.json")

CLASS_NAMES  = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
RISK_LEVELS  = ["Low", "Low-Moderate", "Moderate", "High", "Critical"]
RECOMMENDATIONS = [
    "Routine annual screening. Maintain good glycemic control and regular ophthalmology check-ups.",
    "Follow-up in 9–12 months. Optimize glycemic control, blood pressure management recommended.",
    "Ophthalmology referral within 3–6 months. Consider laser photocoagulation evaluation.",
    "Urgent ophthalmology referral within 1–3 months. High risk of vision loss without intervention.",
    "Immediate ophthalmology referral required. Laser therapy / anti-VEGF injection may be necessary.",
]

# ── Lazy-loaded model state ──────────────────────────────────
_predictor        = None
_clinical_model   = None
_clinical_scaler  = None
_model_metadata   = None


# ── Model loaders ────────────────────────────────────────────

def get_image_predictor():
    global _predictor
    if _predictor is None:
        if not Path(MODEL_PATH).exists():
            logger.warning(f"Keras model not found at {MODEL_PATH}")
            return None
        try:
            from diabetic_retinopathy_model import DRPredictor
            _predictor = DRPredictor(MODEL_PATH)
            logger.info(f"[✓] EfficientNetB3 model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            return None
    return _predictor


def get_clinical_model():
    global _clinical_model, _clinical_scaler
    if _clinical_model is None:
        if not Path(CLINICAL_PATH).exists() or not Path(SCALER_PATH).exists():
            logger.warning("Clinical model files not found.")
            return None, None
        try:
            import joblib
            _clinical_model  = joblib.load(CLINICAL_PATH)
            _clinical_scaler = joblib.load(SCALER_PATH)
            logger.info("[✓] Clinical Random Forest model loaded.")
        except Exception as e:
            logger.error(f"Failed to load clinical model: {e}")
            return None, None
    return _clinical_model, _clinical_scaler


def get_model_metadata():
    global _model_metadata
    if _model_metadata is None and Path(META_PATH).exists():
        with open(META_PATH) as f:
            _model_metadata = json.load(f)
    return _model_metadata


# ── Grad-CAM helper ─────────────────────────────────────────

def generate_gradcam_b64(predictor, image_bytes: bytes) -> Optional[str]:
    """Generate Grad-CAM heatmap and return as base64 PNG."""
    try:
        import cv2
        import tensorflow as tf
        from diabetic_retinopathy_model import get_augmentation_pipeline

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = predictor.preprocessor.preprocess_array(img_bgr)

        IMG_SIZE = 224
        model = predictor.model
        aug = get_augmentation_pipeline(is_training=False)
        img_uint8 = (img * 255).astype(np.uint8)
        aug_result = aug(image=img_uint8)
        img_proc = aug_result["image"].astype(np.float32)

        # Find last conv/activation layer
        target_layer = None
        for layer in reversed(model.layers):
            if any(k in layer.name for k in ("conv", "activation", "top_activation")):
                target_layer = layer
                break
        if target_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        img_tensor = tf.cast(np.expand_dims(img_proc, 0), tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor)
            pred_class = tf.argmax(preds[0])
            class_score = preds[:, pred_class]

        grads  = tape.gradient(class_score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
        cam    = tf.nn.relu(cam).numpy()
        cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        original = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay  = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)

        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        return None


# ── Clinical prediction helper ───────────────────────────────

VA_MAPPING = {
    "20/20": 10, "20/25": 9, "20/30": 8, "20/40": 7,
    "20/50": 6, "20/60": 5, "20/80": 4, "20/100": 3,
    "20/200": 2, "CF": 1, "HM": 0, "LP": 0, "NLP": 0
}

GENDER_MAP    = {"Male": 0, "Female": 1, "Other": 2}
ETHNICITY_MAP = {"Asian": 0, "Black": 1, "Hispanic": 2, "White": 3, "Other": 4}
DTYPE_MAP     = {"Type 1": 0, "Type 2": 1}
SMOKING_MAP   = {"Never": 0, "Former": 1, "Current": 2}
BOOL_MAP      = {"No": 0, "Yes": 1}
EYESIDE_MAP   = {"Right": 0, "Left": 1, "Both": 2}


class ClinicalInput(BaseModel):
    age:                float = Field(..., ge=0, le=120,   description="Patient age in years")
    diabetes_duration:  float = Field(..., ge=0, le=80,    description="Years with diabetes")
    hba1c:              float = Field(..., ge=3.0, le=20.0, description="HbA1c percentage")
    systolic_bp:        float = Field(..., ge=60,  le=300,  description="Systolic blood pressure mmHg")
    diastolic_bp:       float = Field(..., ge=40,  le=200,  description="Diastolic blood pressure mmHg")
    bmi:                float = Field(..., ge=10,  le=70,   description="Body mass index")
    cholesterol:        float = Field(..., ge=50,  le=500,  description="Cholesterol mg/dL")
    iop:                float = Field(..., ge=5,   le=50,   description="Intraocular pressure mmHg")
    va_right:           str   = Field(..., description="Visual acuity right eye e.g. 20/20")
    va_left:            str   = Field(..., description="Visual acuity left eye e.g. 20/20")
    gender:             str   = Field(..., description="Male / Female / Other")
    ethnicity:          str   = Field(..., description="Asian / Black / Hispanic / White / Other")
    diabetes_type:      str   = Field(..., description="Type 1 / Type 2")
    smoking:            str   = Field(..., description="Never / Former / Current")
    kidney_disease:     str   = Field(..., description="Yes / No")
    neuropathy:         str   = Field(..., description="Yes / No")
    eye_side:           str   = Field(..., description="Right / Left / Both")
    has_symptom:        bool  = Field(False, description="Any visual symptoms present")


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health():
    image_model_ready    = Path(MODEL_PATH).exists()
    clinical_model_ready = Path(CLINICAL_PATH).exists() and Path(SCALER_PATH).exists()
    return {
        "status":               "ok",
        "image_model_ready":    image_model_ready,
        "clinical_model_ready": clinical_model_ready,
        "model_paths": {
            "image":    MODEL_PATH,
            "clinical": CLINICAL_PATH,
        },
    }


@app.get("/info")
async def info():
    return {
        "image_model":      "EfficientNetB3 (TensorFlow / Keras)",
        "clinical_model":   "Random Forest (scikit-learn)",
        "num_classes":      5,
        "class_names":      CLASS_NAMES,
        "input_size":       224,
        "dataset":          "APTOS 2019 / EyePACS compatible (image), Clinical dataset 300 patients (tabular)",
        "version":          "2.0.0",
    }


@app.get("/model-metadata")
async def model_metadata():
    meta = get_model_metadata()
    if meta is None:
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return meta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a retinal fundus image.
    Returns DR grade, probabilities, risk level, recommendation, and Grad-CAM heatmap.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB).")

    start     = time.time()
    predictor = get_image_predictor()

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="EfficientNetB3 model not loaded. Ensure dr_model_best.keras exists in backend/models/."
        )

    try:
        result = predictor.predict_from_bytes(image_bytes, tta_rounds=3)
    except Exception as e:
        logger.error(f"Image inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Grad-CAM
    gradcam_b64 = generate_gradcam_b64(predictor, image_bytes)

    grade = result["grade"]
    return JSONResponse(content={
        "grade":           grade,
        "class_name":      result["class_name"],
        "confidence":      result["confidence"],
        "probabilities":   result["probabilities"],
        "risk_level":      RISK_LEVELS[grade],
        "recommendation":  RECOMMENDATIONS[grade],
        "processing_time": round(time.time() - start, 3),
        "filename":        file.filename,
        "gradcam_b64":     gradcam_b64,
        "model":           "EfficientNetB3",
        "demo_mode":       False,
    })


@app.post("/predict-clinical")
async def predict_clinical(data: ClinicalInput):
    """
    Accepts 24 clinical features and returns DR grade prediction from Random Forest model.
    """
    model, scaler = get_clinical_model()
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Clinical model not loaded. Ensure dr_clinical_model.joblib and dr_scaler.joblib exist."
        )

    start = time.time()

    try:
        va_right = VA_MAPPING.get(data.va_right, 5)
        va_left  = VA_MAPPING.get(data.va_left, 5)
        va_avg   = (va_right + va_left) / 2
        va_diff  = abs(va_right - va_left)
        map_val  = data.diastolic_bp + (data.systolic_bp - data.diastolic_bp) / 3
        pulse_p  = data.systolic_bp - data.diastolic_bp
        hba1c_dur = data.hba1c * data.diabetes_duration
        age_dur  = data.age / (data.diabetes_duration + 1)

        features = np.array([[
            data.age,
            data.diabetes_duration,
            data.hba1c,
            data.systolic_bp,
            data.diastolic_bp,
            data.bmi,
            data.cholesterol,
            data.iop,
            va_right,
            va_left,
            va_avg,
            va_diff,
            map_val,
            pulse_p,
            hba1c_dur,
            age_dur,
            GENDER_MAP.get(data.gender, 2),
            ETHNICITY_MAP.get(data.ethnicity, 4),
            DTYPE_MAP.get(data.diabetes_type, 1),
            SMOKING_MAP.get(data.smoking, 0),
            BOOL_MAP.get(data.kidney_disease, 0),
            BOOL_MAP.get(data.neuropathy, 0),
            EYESIDE_MAP.get(data.eye_side, 2),
            int(data.has_symptom),
        ]])

        features_scaled = scaler.transform(features)
        grade  = int(model.predict(features_scaled)[0])
        probs  = model.predict_proba(features_scaled)[0]

        prob_dict = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

        # Risk factor analysis
        risk_factors = []
        if data.hba1c >= 8.0:
            risk_factors.append("High HbA1c (poor glycemic control)")
        if data.diabetes_duration >= 10:
            risk_factors.append(f"Long diabetes duration ({data.diabetes_duration:.0f} years)")
        if data.systolic_bp >= 140:
            risk_factors.append("Hypertension (systolic ≥ 140 mmHg)")
        if data.kidney_disease == "Yes":
            risk_factors.append("Comorbid kidney disease")
        if data.neuropathy == "Yes":
            risk_factors.append("Diabetic neuropathy present")
        if data.smoking == "Current":
            risk_factors.append("Current smoker")

        return JSONResponse(content={
            "grade":           grade,
            "class_name":      CLASS_NAMES[grade],
            "confidence":      round(float(probs[grade]) * 100, 2),
            "probabilities":   prob_dict,
            "risk_level":      RISK_LEVELS[grade],
            "recommendation":  RECOMMENDATIONS[grade],
            "risk_factors":    risk_factors,
            "processing_time": round(time.time() - start, 3),
            "model":           "Random Forest (Clinical Features)",
            "demo_mode":       False,
        })

    except Exception as e:
        logger.error(f"Clinical inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Clinical inference failed: {str(e)}")
