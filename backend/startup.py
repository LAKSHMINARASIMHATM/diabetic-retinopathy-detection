#!/usr/bin/env python3
"""
startup.py — Downloads model weights from environment-variable URLs
before starting the FastAPI server on Render / any cloud host.

Set these env vars in your Render dashboard:
  MODEL_IMAGE_URL   = https://your-host/dr_model_best.keras
  MODEL_CLINICAL_URL = https://your-host/dr_clinical_model.joblib
  MODEL_SCALER_URL  = https://your-host/dr_scaler.joblib

If URLs are not set, the server starts anyway (models loaded lazily on first request).
"""

import os
import sys
import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DOWNLOADS = [
    ("MODEL_IMAGE_URL",    "dr_model_best.keras"),
    ("MODEL_CLINICAL_URL", "dr_clinical_model.joblib"),
    ("MODEL_SCALER_URL",   "dr_scaler.joblib"),
]


def download(url: str, dest: Path):
    import urllib.request, shutil
    print(f"⬇️  Downloading {dest.name} from {url[:60]}...")
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    print(f"✅  Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for env_var, filename in DOWNLOADS:
        url = os.environ.get(env_var)
        dest = MODELS_DIR / filename
        if url and not dest.exists():
            try:
                download(url, dest)
            except Exception as e:
                print(f"⚠️  Could not download {filename}: {e}")
        elif dest.exists():
            print(f"✔️  {filename} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"ℹ️  {env_var} not set — skipping {filename}")

    # Hand off to uvicorn
    port = os.environ.get("PORT", "8000")
    print(f"\n🚀 Starting FastAPI on port {port}...")
    os.execvp("uvicorn", ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", port])
