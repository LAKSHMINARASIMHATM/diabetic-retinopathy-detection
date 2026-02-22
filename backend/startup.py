#!/usr/bin/env python3
"""
startup.py — Downloads model weights from HuggingFace then starts FastAPI.

Models are hosted at: https://huggingface.co/narasimha01tm/retinopath
Override URLs via environment variables if needed:
  MODEL_IMAGE_URL    — URL for dr_model_best.keras
  MODEL_CLINICAL_URL — URL for dr_clinical_model.joblib
  MODEL_SCALER_URL   — URL for dr_scaler.joblib
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

HF_BASE = "https://huggingface.co/narasimha01tm/retinopath/resolve/main"

DOWNLOADS = [
    ("MODEL_IMAGE_URL",    "dr_model_best.keras",      f"{HF_BASE}/dr_model_best.keras"),
    ("MODEL_CLINICAL_URL", "dr_clinical_model.joblib", f"{HF_BASE}/dr_clinical_model.joblib"),
    ("MODEL_SCALER_URL",   "dr_scaler.joblib",         f"{HF_BASE}/dr_scaler.joblib"),
]


def download(url: str, dest: Path):
    import urllib.request, shutil
    print(f"⬇️  Downloading {dest.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=300) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    print(f"✅  Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    for env_var, filename, default_url in DOWNLOADS:
        url  = os.environ.get(env_var, default_url)
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"✔️  {filename} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            try:
                download(url, dest)
            except Exception as e:
                print(f"⚠️  Could not download {filename}: {e}")

    # Hand off to uvicorn
    port = os.environ.get("PORT", "7860")
    print(f"\n🚀 Starting FastAPI on port {port}...", flush=True)
    os.execvp("uvicorn", ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", port])
