# 👁️ RetinaAI — Diabetic Retinopathy Detection System

<div align="center">

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**A dual-AI engine for Diabetic Retinopathy screening — combining deep learning on fundus images with ensemble machine learning on clinical features.**

[🔬 Image Analysis](#-image-analysis-page) · [🩺 Clinical Assessment](#-clinical-assessment-page) · [📋 Prescription Generator](#-prescription-generator) · [📖 About](#-about)

</div>

---

## 📸 Overview

RetinaAI provides two independent, complementary pathways to detect and grade Diabetic Retinopathy (DR) severity across the **International Clinical DR Scale (ICDR)** — Grade 0 (No DR) through Grade 4 (Proliferative DR):

| Pathway | Model | Input | Output |
|---|---|---|---|
| **Image Analysis** | EfficientNetB3 (Deep Learning) | Fundus photograph | DR grade + Grad-CAM heatmap |
| **Clinical Assessment** | Random Forest (Ensemble ML) | 24 clinical features | DR grade + risk factors |

Both models are trained on real-world data and served via a **FastAPI** backend, with a **Next.js 15** frontend.

---

## 🧠 Model Architecture

### 🔬 EfficientNetB3 — Image Model
- **Base:** EfficientNetB3 pre-trained on ImageNet (5.3M parameters)
- **Fine-tuning:** Two-stage (head-only → full) on fundus image dataset
- **Input:** 224×224 RGB retinal fundus photograph
- **Augmentation:** CLAHE, flips, rotations, brightness shifts, Gaussian noise
- **TTA:** 3-round Test-Time Augmentation for robust predictions
- **Explainability:** Grad-CAM heatmap overlay highlighting retinal lesions
- **Output:** 5-class softmax (No DR / Mild / Moderate / Severe / Proliferative)

### 🌲 Random Forest — Clinical Model
- **Algorithm:** Random Forest Classifier (500 estimators, max depth 12)
- **Features:** 24 engineered features from patient clinical data
- **Training:** 240 patients with 5-fold stratified cross-validation
- **Class balancing:** `class_weight='balanced'`
- **Features include:** Age, HbA1c, BP, BMI, IOP, visual acuity, diabetes duration, comorbidities, and derived features (MAP, pulse pressure, HbA1c × duration)

---

## 🗂️ Project Structure

```
diabetic-retinopathy-detection/
│
├── backend/                        # FastAPI Python backend
│   ├── server.py                   # Main API server (endpoints)
│   ├── diabetic_retinopathy_model.py  # EfficientNetB3 + Grad-CAM class
│   ├── train_clinical_model.py     # Random Forest training script
│   ├── requirements.txt            # Python dependencies
│   └── models/                     # Model weights (not in git — see below)
│       ├── dr_model_best.keras     # EfficientNetB3 weights (~55MB)
│       ├── dr_clinical_model.joblib  # Random Forest model
│       ├── dr_scaler.joblib        # Feature scaler
│       └── dr_model_metadata.json  # Feature names, metrics, class info
│
├── app/                            # Next.js App Router pages
│   ├── page.tsx                    # Home / Landing page
│   ├── analyze/page.tsx            # Image Analysis page
│   ├── clinical/page.tsx           # Clinical Assessment form
│   ├── report/page.tsx             # Prescription Generator
│   ├── about/page.tsx              # Model info & ICDR scale
│   ├── layout.tsx                  # Root layout + metadata
│   └── globals.css                 # Global styles + animations
│
├── components/
│   └── Navbar.tsx                  # Shared navigation bar
│
├── public/                         # Static assets
├── package.json
├── tsconfig.json
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10–3.12**
- **Node.js 18+** and **npm**
- **Git**

---

### 1️⃣ Clone the repository

```bash
git clone https://github.com/LAKSHMINARASIMHATM/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

---

### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### ⚠️ Download Model Weights

The trained model weights are too large for GitHub. Download them separately:

| File | Description | Size |
|---|---|---|
| `dr_model_best.keras` | EfficientNetB3 weights | ~55 MB |
| `dr_clinical_model.joblib` | Random Forest model | ~5 MB |
| `dr_scaler.joblib` | StandardScaler | <1 MB |

Place them in `backend/models/`.

**Or retrain from scratch:**
```bash
# Train the clinical Random Forest model
python train_clinical_model.py

# Train the image model (requires APTOS/EyePACS dataset)
# See diabetic_retinopathy_model.py for training setup
```

#### ▶️ Start the Backend

```bash
uvicorn server:app --reload --port 8000
```

API docs available at: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

### 3️⃣ Frontend Setup

```bash
# From the project root
npm install
npm run dev
```

App available at: **[http://localhost:3000](http://localhost:3000)**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — model load status |
| `POST` | `/predict` | Image inference (multipart/form-data) |
| `POST` | `/predict-clinical` | Clinical inference (JSON body) |
| `GET` | `/model-metadata` | Clinical model feature info & metrics |
| `GET` | `/docs` | Interactive Swagger UI |

### Example — Image Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@retinal_image.jpg"
```

### Example — Clinical Prediction
```bash
curl -X POST http://localhost:8000/predict-clinical \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "diabetes_duration": 10,
    "hba1c": 8.2,
    "systolic_bp": 140,
    "diastolic_bp": 88,
    "bmi": 28.5,
    "cholesterol": 210,
    "iop": 17,
    "va_right": "20/40",
    "va_left": "20/30",
    "gender": "Male",
    "ethnicity": "Asian",
    "diabetes_type": "Type 2",
    "smoking": "Former",
    "kidney_disease": "No",
    "neuropathy": "Yes",
    "eye_side": "Both",
    "has_symptom": true
  }'
```

---

## 📄 Pages

### 🏠 Home
Landing page with feature overview, DR severity scale, model statistics, and links to all tools.

### 🔬 Image Analysis Page
- Drag-and-drop or click-to-upload retinal fundus photograph
- Real-time inference with EfficientNetB3
- **Grad-CAM heatmap** toggle — visualizes model attention on retinal lesions
- Per-class probability bars with severity color coding
- Clinical recommendation based on predicted grade

### 🩺 Clinical Assessment Page
- Structured form with 5 sections: Demographics, Diabetes Info, Vital Signs, Ophthalmic, Comorbidities
- 24 features fed to the Random Forest model
- Identified risk factors displayed with probability breakdown
- Follow-up recommendation

### 📋 Prescription Generator
- Enter patient name, age, gender, MRN, doctor, hospital
- Select DR grade from AI results
- Generates a **printable medical prescription** with:
  - Diagnosis summary
  - Medications with dosage & purpose
  - Eye drops / intravitreal injections (for severe grades)
  - Investigations advised
  - Lifestyle & dietary advice
  - Auto-calculated follow-up date
- **Print / Save as PDF** via browser

### ℹ️ About
- EfficientNetB3 & Random Forest architecture comparison
- 6-step image analysis pipeline diagram
- 24 clinical features breakdown
- Full ICDR grading reference (Grade 0–4) with clinical features and recommended actions

---

## 🎨 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 15, React 19, TypeScript |
| **Styling** | Vanilla CSS, Tailwind CSS v4, Glassmorphism |
| **Icons** | Lucide React |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Image Model** | TensorFlow/Keras, EfficientNetB3 |
| **Clinical Model** | scikit-learn, Random Forest |
| **Image Processing** | OpenCV, Albumentations, Pillow |
| **Explainability** | Grad-CAM (gradient-weighted class activation maps) |

---

## 📊 DR Severity Scale (ICDR)

| Grade | Name | Key Features | Action |
|---|---|---|---|
| **0** | No DR | No abnormalities | Annual screening |
| **1** | Mild NPDR | Microaneurysms only | Recheck in 12 months |
| **2** | Moderate NPDR | Hemorrhages, exudates, cotton wool spots | Referral in 3–6 months |
| **3** | Severe NPDR | 4-2-1 rule: extensive hemorrhages, venous beading, IRMA | Urgent referral 1–3 months |
| **4** | Proliferative DR | Neovascularization, vitreous hemorrhage | **Immediate** laser/anti-VEGF |

---

## ⚠️ Disclaimer

> RetinaAI is intended for **educational and screening assistance purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment by a licensed ophthalmologist or physician. Always consult a qualified healthcare professional for medical decisions.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by [LAKSHMINARASIMHATM](https://github.com/LAKSHMINARASIMHATM)

⭐ Star this repo if you find it useful!

</div>
