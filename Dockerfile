# ==============================================================================
# VisionInspect - Production Multi-Stage Hardened Dockerfile
# Base: Python 3.12 Slim (Minimal Debian Base)
# ==============================================================================

FROM python:3.12-slim AS builder

WORKDIR /app

# Install minimal OS build dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-specific PyTorch wheels to minimize container footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Final Production Runtime Stage
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install minimal runtime shared libraries for OpenCV and headless graphics
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create dedicated unprivileged non-root user and group
RUN groupadd -g 10001 visiongroup && \
    useradd -u 10001 -g visiongroup -s /bin/bash -m visionuser

# Copy application source code and models
COPY --chown=visionuser:visiongroup app/ ./app/
COPY --chown=visionuser:visiongroup src/ ./src/
COPY --chown=visionuser:visiongroup configs/ ./configs/
COPY --chown=visionuser:visiongroup models/ ./models/
COPY --chown=visionuser:visiongroup assets/ ./assets/

# Restrict model files to read-only for runtime security
RUN chmod -R 550 ./models

# Set secure environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISIONINSPECT_ENV=production \
    VISIONINSPECT_DEVICE=cpu \
    KMP_DUPLICATE_LIB_OK=TRUE

# Switch to non-root user
USER visionuser

# Expose Streamlit frontend (8501) and FastAPI backend (8000)
EXPOSE 8000 8501

# Default healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production entrypoint
CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]
