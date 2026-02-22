# ── Base image: Python 3.12 slim ──────────────────────────────
FROM python:3.12-slim

# HuggingFace Spaces requires port 7860
ENV PORT=7860

# Suppress TF logs noise
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# OpenCV needs these libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── App directory ─────────────────────────────────────────────
WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Create models dir (models downloaded at runtime via startup.py)
RUN mkdir -p models

# Expose HF Spaces port
EXPOSE 7860

# startup.py downloads models from HF then launches uvicorn
CMD ["python", "startup.py"]
