---
title: RetinoPath Backend
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# RetinoPath — Diabetic Retinopathy Detection API

FastAPI backend serving:
- **EfficientNetB3** model for fundus image grading
- **Random Forest** model for clinical feature assessment

## Endpoints
- `GET /health` — Health check
- `GET /docs` — Interactive Swagger UI
- `POST /predict` — Fundus image → DR grade + Grad-CAM
- `POST /predict-clinical` — Clinical features → DR grade
