# ==============================================================================
# VisionInspect - Production Multi-Stage Hardened Dockerfile
# Base: Python 3.12 Slim (Minimal Debian Base)
# Architecture: FastAPI REST API + Streamlit Web UI coordinated via Nginx on $PORT
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

# Install runtime dependencies: Nginx, gettext (for envsubst), graphics & curl
RUN apt-get update && apt-get install --no-install-recommends -y \
    nginx \
    gettext-base \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create dedicated unprivileged non-root user and group
RUN groupadd -g 10001 visiongroup && \
    useradd -u 10001 -g visiongroup -s /bin/bash -m visionuser

# Copy application source code, models, configs, nginx, and entrypoint
COPY --chown=visionuser:visiongroup app/ ./app/
COPY --chown=visionuser:visiongroup src/ ./src/
COPY --chown=visionuser:visiongroup configs/ ./configs/
COPY --chown=visionuser:visiongroup models/ ./models/
COPY --chown=visionuser:visiongroup assets/ ./assets/
COPY --chown=visionuser:visiongroup nginx/ ./nginx/
COPY --chown=visionuser:visiongroup scripts/entrypoint.sh ./scripts/entrypoint.sh

# Restrict model files to read-only for runtime security & make entrypoint executable
RUN sed -i 's/\r$//' ./scripts/entrypoint.sh && \
    chmod -R 550 ./models && \
    chmod +x ./scripts/entrypoint.sh && \
    chown -R visionuser:visiongroup /var/log/nginx /var/lib/nginx /etc/nginx

# Set secure environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISIONINSPECT_ENV=production \
    VISIONINSPECT_DEVICE=cpu \
    VISIONINSPECT_BACKEND_URL=http://127.0.0.1:8000 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=10000

# Switch to non-root user
USER visionuser

# Expose Render service port
EXPOSE 10000

# Default healthcheck probing Nginx /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch production entrypoint
CMD ["/app/scripts/entrypoint.sh"]
