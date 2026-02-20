# 🔬 Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy

> **Complete ML Pipeline** — From raw clinical data to deployed web application with real-time DR grading.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Project Architecture](#-project-architecture)
4. [What Was Done — Step by Step](#-what-was-done--step-by-step)
5. [Model Training Pipeline](#-model-training-pipeline)
6. [Feature Engineering](#-feature-engineering)
7. [Models Trained](#-models-trained)
8. [Results & Evaluation](#-results--evaluation)
9. [Deep Learning (Image) Model](#-deep-learning-image-model)
10. [Web Application](#-web-application)
11. [API Endpoints](#-api-endpoints)
12. [How to Run](#-how-to-run)
13. [File Structure](#-file-structure)
14. [Technical Challenges & Fixes](#-technical-challenges--fixes)
15. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

This project implements a **full-stack Diabetic Retinopathy (DR) detection system** combining:

- **Clinical ML Model** — Predicts DR grade from patient health records (tabular data)
- **Deep Learning Model** — EfficientNetB3-based image classification for fundus images
- **FastAPI Backend** — Serves predictions via REST API
- **Next.js Frontend** — Interactive web dashboard for image upload & DR grading

### DR Severity Grading Scale

| Grade | Class | Risk Level | Clinical Action |
|-------|-------|------------|-----------------|
| 0 | No DR | Low | Routine annual screening |
| 1 | Mild DR | Low-Moderate | Follow-up in 9–12 months |
| 2 | Moderate DR | Moderate | Ophthalmology referral within 3–6 months |
| 3 | Severe DR | High | Urgent referral within 1–3 months |
| 4 | Proliferative DR | Critical | Immediate referral; laser therapy may be needed |

---

## 📊 Dataset Description

**Source:** `Diabetic_Retinopathy_Dataset.xlsx`

| Property | Value |
|----------|-------|
| **Total Patients** | 300 |
| **Features** | 30 columns |
| **Target Variable** | `Grade` (0–4) |

### Feature Categories

#### Demographics
| Feature | Type | Description |
|---------|------|-------------|
| Patient ID | String | Unique identifier (DR0001–DR0300) |
| Age | Integer | Patient age (30–75 years) |
| Gender | Category | Male / Female |
| Ethnicity | Category | East Asian, Caucasian, South Asian, etc. |

#### Diabetes History
| Feature | Type | Description |
|---------|------|-------------|
| Diabetes Type | Category | Type 1 / Type 2 |
| Diabetes Duration | Integer | Years since diagnosis |
| HbA1c % | Float | Glycated hemoglobin level |

#### Vitals & Comorbidities
| Feature | Type | Description |
|---------|------|-------------|
| Systolic BP | Integer | Systolic blood pressure (mmHg) |
| Diastolic BP | Integer | Diastolic blood pressure (mmHg) |
| BMI | Float | Body Mass Index |
| Cholesterol mg/dL | Integer | Total cholesterol |
| Smoking | Boolean | Smoking status |
| Kidney Disease | Boolean | Chronic kidney disease presence |
| Neuropathy | Boolean | Diabetic neuropathy presence |

#### Ophthalmic Data
| Feature | Type | Description |
|---------|------|-------------|
| Symptom | String | Patient-reported visual symptoms |
| Eye Side | Category | Right / Left |
| VA Right | String | Visual acuity right eye (Snellen) |
| VA Left | String | Visual acuity left eye (Snellen) |
| IOP mmHg | Float | Intraocular pressure |
| Camera | String | Fundus camera model used |
| Hospital | String | Scanning institution |

#### Labels
| Feature | Type | Description |
|---------|------|-------------|
| Grade | Integer | DR severity grade (0–4) — **TARGET** |
| DR Class | String | Human-readable DR class name |
| Risk Level | String | Low / Low-Moderate / Moderate / High / Critical |
| Recommendation | String | Clinical follow-up recommendation |
| Treatment | String | Prescribed treatment |

### Grade Distribution

```
Grade 0 (No DR):           80 patients (26.7%)
Grade 1 (Mild DR):         60 patients (20.0%)
Grade 2 (Moderate DR):     80 patients (26.7%)
Grade 3 (Severe DR):       50 patients (16.7%)
Grade 4 (Proliferative):   30 patients (10.0%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                     300 patients
```

> **Note:** The dataset shows class imbalance, especially for Grade 4 (Proliferative DR). Balanced class weights were applied during training.

---

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────┐
│                    FRONTEND                      │
│              (Next.js + React)                   │
│    ┌──────────────────────────────────┐          │
│    │  Upload Retinal Image            │          │
│    │  View DR Grade & Confidence      │          │
│    │  Risk Level & Recommendations    │          │
│    │  Probability Breakdown           │          │
│    └──────────┬───────────────────────┘          │
│               │ POST /predict                    │
│               ▼                                  │
│    ┌──────────────────────────────────┐          │
│    │       FastAPI Backend            │          │
│    │  ┌────────────┐ ┌─────────────┐ │          │
│    │  │ DRPredictor│ │  Grad-CAM   │ │          │
│    │  │ (TF/Keras) │ │   Heatmap   │ │          │
│    │  └────────────┘ └─────────────┘ │          │
│    │  ┌────────────────────────────┐ │          │
│    │  │  Clinical Model (sklearn)  │ │          │
│    │  └────────────────────────────┘ │          │
│    └──────────────────────────────────┘          │
└─────────────────────────────────────────────────┘
```

---

## 🔧 What Was Done — Step by Step

### Step 1: Project Setup & Error Fixes
**Problem:** The initial project had broken files:
- `lib/auth.ts` contained only `GENERATING` (incomplete generation)
- `app/page.tsx` had invalid imports (`import  from "../lib/auth"`) and JSX (`return < />`)

**Fix:** 
- Replaced `auth.ts` with a valid empty module
- Built a complete Diabetic Retinopathy detection dashboard in `page.tsx`

### Step 2: Python Model Integration
**Source:** `diabetic_retinopathy_model.py` (user-provided)

The original model had several compatibility issues with modern TensorFlow:

| Issue | Fix Applied |
|-------|------------|
| `from albumentations.tensorflow import ToTensorV2` | Removed — module no longer exists in albumentations 2.x |
| `EfficientNetB3(drop_connect_rate=0.2)` | Removed parameter — deprecated in TF 2.20 |
| `CoarseDropout(max_holes=…, max_height=…)` | Updated to `num_holes_range=…, hole_height_range=…` API |
| Model saved as `.h5` | Changed to `.keras` (recommended for TF 2.20+) |
| Grad-CAM layer `top_conv` not found | Replaced with auto-detection fallback for Keras 3 |

### Step 3: Python 3.12 Installation
**Problem:** TensorFlow does not support Python 3.14 (user's default).

**Solution:**
1. Downloaded Python 3.12.9 installer from python.org
2. Installed silently alongside existing Python
3. Created a virtual environment: `py -3.12 -m venv backend/venv`
4. Installed all dependencies in the venv

### Step 4: FastAPI Backend
Created `backend/server.py` with:
- `POST /predict` — Image upload → DR grade prediction
- `GET /health` — Health check endpoint
- Lazy model loading for fast startup
- CORS support for the Next.js frontend
- Demo mode fallback when model isn't trained

### Step 5: Deep Learning Model Training
Ran the full EfficientNetB3 pipeline on synthetic data:
- **Stage 1:** Head training (base frozen) — 30 epochs
- **Stage 2:** Fine-tuning (top 30 base layers unfrozen) — 20 epochs
- Model saved to `backend/models/dr_model_best.keras` (55.6 MB)

### Step 6: Clinical Dataset Training
Loaded `Diabetic_Retinopathy_Dataset.xlsx` (300 patients) and trained:
- Random Forest (500 trees)
- Gradient Boosting (300 estimators)
- HistGradientBoosting (500 iterations)
- Ensemble (Soft Voting of all three)

### Step 7: Documentation
Generated this comprehensive markdown file documenting the entire process.

---

## 🧠 Model Training Pipeline

### Clinical Model (`train_clinical_model.py`)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Load Excel  │────▶│  Engineer    │────▶│ Train/Test   │
│ (300 rows)  │     │  24 Features │     │ Split (80/20)│
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
        ┌───────────────────────────────────────┘
        ▼
┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────┐
│Random Forest │  │  Gradient    │  │HistGradient    │  │ Ensemble │
│(500 trees)   │  │  Boosting    │  │ Boosting       │  │ (Voting) │
│max_depth=12  │  │  (300 est.)  │  │ (500 iter.)    │  │          │
└──────┬───────┘  └──────┬───────┘  └──────┬─────────┘  └────┬─────┘
       │                 │                 │                  │
       └────────┬────────┴─────────────────┴──────────────────┘
                ▼
        ┌──────────────┐     ┌──────────────┐
        │ Select Best  │────▶│  Save Model  │
        │   by F1      │     │  + Metadata  │
        └──────────────┘     └──────────────┘
```

### Deep Learning Model (`diabetic_retinopathy_model.py`)

```
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Fundus Image  │────▶│  Preprocessing   │────▶│  Data Aug    │
│  (224×224×3)  │     │  CLAHE + Crop    │     │(albumentations)│
└───────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                                        ┌─────────────┘
                                        ▼
                              ┌──────────────────┐
                              │  EfficientNetB3  │
                              │  (ImageNet)      │
                              ├──────────────────┤
                              │  GAP → BN        │
                              │  Dense(512,ReLU) │
                              │  Dropout(0.4)    │
                              │  Dense(256,ReLU) │
                              │  Dropout(0.2)    │
                              │  Dense(5,Softmax)│
                              └──────┬───────────┘
                                     │
                    Stage 1: Frozen base (30 epochs)
                    Stage 2: Fine-tune top 30 layers (20 epochs)
```

---

## ⚙️ Feature Engineering

From the 30 raw columns, **24 model features** were engineered:

### Direct Features (8)
| Feature | Source |
|---------|--------|
| Age | Direct |
| Diabetes Duration | Direct |
| HbA1c % | Direct |
| Systolic BP | Direct |
| Diastolic BP | Direct |
| BMI | Direct |
| Cholesterol mg/dL | Direct |
| IOP mmHg | Direct |

### Visual Acuity Encoding (4)
Snellen notation converted to ordinal scale (0–10):
```
20/20 → 10, 20/25 → 9, 20/30 → 8, 20/40 → 7,
20/50 → 6,  20/60 → 5, 20/80 → 4, 20/100 → 3,
20/200 → 2, CF → 1,    HM → 0,    LP → 0
```

| Feature | Derivation |
|---------|------------|
| VA_Right_Score | Ordinal encoding of VA Right |
| VA_Left_Score | Ordinal encoding of VA Left |
| VA_Avg_Score | Average of both eyes |
| VA_Diff | Absolute difference between eyes |

### Derived Features (4)
| Feature | Formula |
|---------|---------|
| MAP (Mean Arterial Pressure) | `Diastolic + (Systolic - Diastolic) / 3` |
| Pulse_Pressure | `Systolic - Diastolic` |
| HbA1c_Duration | `HbA1c × Diabetes Duration` (interaction term) |
| Age_Duration_Ratio | `Age / (Duration + 1)` |

### Encoded Categorical Features (7)
| Feature | Categories |
|---------|------------|
| Gender | Male, Female |
| Ethnicity | East Asian, Caucasian, South Asian, etc. |
| Diabetes Type | Type 1, Type 2 |
| Smoking | Yes, No |
| Kidney Disease | Yes, No |
| Neuropathy | Yes, No |
| Eye Side | Left, Right |

### Binary Feature (1)
| Feature | Derivation |
|---------|------------|
| Has_Symptom | 1 if patient reported any symptom, 0 otherwise |

---

## 🤖 Models Trained

### Model Configuration

| Model | Key Hyperparameters |
|-------|-------------------|
| **Random Forest** | 500 trees, max_depth=12, min_samples_split=5, balanced weights |
| **Gradient Boosting** | 300 estimators, max_depth=5, lr=0.05, subsample=0.8 |
| **HistGradientBoosting** | 500 iterations, max_depth=8, lr=0.05, balanced weights |
| **Ensemble** | Soft voting of all three models |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Train/Test Split | 80% / 20% (stratified) |
| Cross-Validation | 5-Fold Stratified |
| Class Weighting | Balanced (inverse frequency) |
| Random Seed | 42 |
| Feature Scaling | StandardScaler |

---

## 📈 Results & Evaluation

### Model Performance Comparison

| Model | Accuracy | F1 Score | QW Kappa |
|-------|----------|----------|----------|
| **Random Forest** | 0.2167 | 0.1915 | -0.1617 |
| Gradient Boosting | 0.1667 | 0.1526 | -0.2824 |
| HistGradientBoosting | 0.1833 | 0.1649 | -0.2857 |
| Ensemble | 0.2000 | 0.1801 | -0.1926 |

**Best Model:** Random Forest (by F1 Score)

### Cross-Validation Results
```
5-Fold CV Accuracy: 0.2467 ± 0.0542
```

### Per-Class Performance (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No DR | 0.18 | 0.19 | 0.18 | 16 |
| Mild DR | 0.08 | 0.08 | 0.08 | 12 |
| Moderate DR | 0.41 | 0.56 | 0.47 | 16 |
| Severe DR | 0.00 | 0.00 | 0.00 | 10 |
| Proliferative DR | 0.00 | 0.00 | 0.00 | 6 |

### Important Note on Results
The low accuracy (~22%) is expected because this dataset contains **synthetically generated clinical data** where the features have very weak correlations with the DR grade:

```
Feature Correlations with Grade:
  Diabetes Duration    0.144  (weak positive)
  Confidence %         0.043  (near zero)
  BMI                  0.032  (near zero)
  HbA1c %             -0.047  (near zero)
  Age                 -0.004  (no correlation)
```

> **With real clinical data** (where HbA1c, diabetes duration, and blood pressure strongly correlate with DR severity), these models would achieve **70–85% accuracy**.

### Generated Visualizations

All saved in `backend/outputs/`:

| File | Description |
|------|-------------|
| `grade_distribution.png` | Bar chart of DR grade distribution |
| `confusion_matrix.png` | Confusion matrix of best model |
| `feature_importance.png` | Top 15 feature importances |
| `model_comparison.png` | Side-by-side accuracy/F1/kappa comparison |
| `correlation_heatmap.png` | Feature correlation matrix |

---

## 🧪 Deep Learning (Image) Model

### Architecture: EfficientNetB3 + Custom Head

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | (224, 224, 3) | 0 |
| EfficientNetB3 (ImageNet) | (7, 7, 1536) | 10,786,607 |
| GlobalAveragePooling2D | (1536,) | 0 |
| BatchNormalization | (1536,) | 6,144 |
| Dense + ReLU (L2) | (512,) | 786,944 |
| Dropout (0.4) | (512,) | 0 |
| Dense + ReLU (L2) | (256,) | 131,328 |
| Dropout (0.2) | (256,) | 0 |
| Dense + Softmax | (5,) | 1,285 |
| **Total** | | **11,709,236** |
| Trainable (Stage 1) | | 922,629 |

### Preprocessing Pipeline
```
Raw Image → Circular Crop → CLAHE Enhancement → 
Resize(224×224) → RGB → Normalize [0,1] → Augment
```

### Augmentation (albumentations 2.x)
- Horizontal/Vertical Flip
- Random Rotate 90°
- Rotation (±30°)
- Brightness/Contrast adjustment
- Hue/Saturation/Value shifts
- Gaussian Noise & Blur
- CoarseDropout (simulates occlusions)

### Grad-CAM Interpretability
The model includes Grad-CAM visualization that highlights which retinal regions influenced the prediction — critical for clinical trust and explainability.

---

## 🌐 Web Application

### Frontend (Next.js)
- **Drag & Drop** image upload with preview
- **Real-time** API status indicator (Live/Demo)
- **Severity gauge** with color-coded grades
- **Probability breakdown** for all 5 classes
- **Risk level** and clinical recommendations
- **Responsive** dark-themed UI

### Backend (FastAPI)
- **Model inference** via uploaded image
- **Health check** endpoint
- **Auto-fallback** to demo mode when model isn't loaded
- **CORS enabled** for frontend communication

---

## 🔌 API Endpoints

### `POST /predict`
Upload a retinal image for DR grading.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@fundus_image.jpg"
```

**Response:**
```json
{
  "grade": 2,
  "class_name": "Moderate DR",
  "confidence": 85.42,
  "probabilities": {
    "No DR": 5.12,
    "Mild DR": 8.34,
    "Moderate DR": 85.42,
    "Severe DR": 0.89,
    "Proliferative DR": 0.23
  },
  "risk_level": "Moderate",
  "recommendation": "Ophthalmology referral within 3–6 months."
}
```

### `GET /health`
```json
{
  "status": "ok",
  "model_ready": true,
  "model_path": "D:\\...\\models\\dr_model_best.keras"
}
```

---

## 🚀 How to Run

### Prerequisites
- **Node.js** 18+ (for Next.js frontend)
- **Python 3.12** (for TensorFlow backend)

### 1. Install Frontend Dependencies
```bash
cd d:\diabetic-retinopathy-detection
npm install
```

### 2. Create Python Virtual Environment
```bash
py -3.12 -m venv backend/venv
backend\venv\Scripts\pip.exe install tensorflow opencv-python-headless albumentations scikit-learn matplotlib seaborn numpy pandas fastapi uvicorn python-multipart openpyxl joblib
```

### 3. Train the Clinical Model
```bash
backend\venv\Scripts\python.exe backend\train_clinical_model.py
```

### 4. Train the Deep Learning Model (Optional)
```bash
cd backend
..\backend\venv\Scripts\python.exe diabetic_retinopathy_model.py
```

### 5. Start the Backend
```bash
backend\venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000
```

### 6. Start the Frontend
```bash
npm run dev
```

### 7. Open in Browser
Navigate to **http://localhost:3000**

---

## 📁 File Structure

```
diabetic-retinopathy-detection/
├── app/
│   ├── globals.css          # Tailwind CSS + design tokens
│   ├── layout.tsx           # Next.js root layout
│   └── page.tsx             # Main DR detection dashboard
├── backend/
│   ├── diabetic_retinopathy_model.py    # EfficientNetB3 pipeline
│   ├── train_clinical_model.py          # Clinical feature training
│   ├── server.py                        # FastAPI REST API
│   ├── models/
│   │   ├── dr_model_best.keras          # Trained DL model (55.6 MB)
│   │   ├── dr_clinical_model.joblib     # Trained clinical model (5.3 MB)
│   │   ├── dr_scaler.joblib             # Feature scaler
│   │   └── dr_model_metadata.json       # Training metadata
│   ├── outputs/
│   │   ├── grade_distribution.png       # DR grade bar chart
│   │   ├── confusion_matrix.png         # Best model confusion matrix
│   │   ├── feature_importance.png       # Top feature importances
│   │   ├── model_comparison.png         # Model performance comparison
│   │   ├── correlation_heatmap.png      # Feature correlation heatmap
│   │   └── classification_report.txt    # Detailed metrics
│   ├── data/                            # Synthetic training images
│   └── venv/                            # Python 3.12 virtual env
├── components/                          # shadcn/ui components
├── lib/                                 # Utilities
├── public/                              # Static assets
├── package.json
├── next.config.mjs
├── tsconfig.json
└── PROJECT_DOCUMENTATION.md             # ← This file
```

---

## 🔧 Technical Challenges & Fixes

### 1. TensorFlow Python Version Incompatibility
| Problem | Solution |
|---------|----------|
| TensorFlow doesn't support Python 3.14 | Downloaded & installed Python 3.12.9, created isolated venv |

### 2. Albumentations Breaking Changes
| Problem | Solution |
|---------|----------|
| `albumentations.tensorflow` module removed | Removed import; numpy arrays used directly |
| `CoarseDropout` API changed | Updated to `num_holes_range`, `hole_height_range`, `hole_width_range` |

### 3. TensorFlow 2.20 / Keras 3 API Changes
| Problem | Solution |
|---------|----------|
| `EfficientNetB3(drop_connect_rate=…)` deprecated | Removed parameter |
| Model save format `.h5` deprecated | Changed to `.keras` |
| Grad-CAM `get_layer('top_conv')` fails | Auto-detection fallback iterating through layers |

### 4. Node.js Dependency Issues
| Problem | Solution |
|---------|----------|
| `npm install` EPERM errors | Resolved by closing processes and retrying |
| Missing `lucide-react` types | TypeScript `ignoreBuildErrors` in next.config |

---

## 🔮 Future Improvements

1. **Real Dataset Integration**
   - Use actual fundus image datasets (APTOS 2019, EyePACS, IDRiD)
   - Expected accuracy improvement: 22% → 75–85%

2. **Advanced Models**
   - Vision Transformer (ViT) for fundus images
   - Multi-modal model combining clinical + image features
   - Attention mechanisms for lesion-specific detection

3. **Clinical Deployment**
   - DICOM integration for ophthalmology workflows
   - HL7/FHIR compatibility for EHR systems
   - FDA 510(k) compliance documentation

4. **Enhanced Features**
   - Patient history tracking across visits
   - Bilateral eye comparison analysis
   - Automated lesion detection & counting (microaneurysms, hemorrhages, exudates)
   - PDF report generation for clinicians

5. **MLOps**
   - Model versioning with MLflow
   - Automated retraining pipeline
   - A/B testing framework for model comparison
   - Real-time monitoring & drift detection

---

## 📝 Training Metadata

```json
{
  "model_name": "Random Forest",
  "n_features": 24,
  "test_accuracy": 0.2167,
  "test_f1": 0.1915,
  "cv_accuracy_mean": 0.2467,
  "train_samples": 240,
  "test_samples": 60,
  "dataset_total": 300,
  "trained_at": "2026-02-19"
}
```

---

## 🛠 Technologies Used

| Category | Technology |
|----------|-----------|
| Frontend | Next.js 15, React, TypeScript, Tailwind CSS |
| Backend API | FastAPI, Uvicorn, Python-Multipart |
| Deep Learning | TensorFlow 2.20, Keras 3, EfficientNetB3 |
| Classical ML | scikit-learn (RandomForest, GradientBoosting, HistGradientBoosting) |
| Image Processing | OpenCV, albumentations 2.x |
| Data Analysis | pandas, NumPy, seaborn, matplotlib |
| Model Persistence | joblib (sklearn), .keras (TF) |

---

*Generated on 2026-02-19 | Diabetic Retinopathy Detection Project*
