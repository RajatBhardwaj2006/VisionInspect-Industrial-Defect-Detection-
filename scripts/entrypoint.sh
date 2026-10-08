#!/bin/bash
set -e

# Default to port 10000 (Render default) if PORT is not set
PORT=${PORT:-10000}
export PORT

echo "===================================================================="
echo "             VISIONINSPECT PRODUCTION SERVICE STARTUP               "
echo "===================================================================="
echo "Public Service Port: ${PORT}"
echo "Active Device:       ${VISIONINSPECT_DEVICE:-cpu}"
echo "Environment:         ${VISIONINSPECT_ENV:-production}"
echo "===================================================================="

# Substitute $PORT into Nginx config
mkdir -p /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp
envsubst '${PORT}' < /app/nginx/nginx.conf.template > /tmp/nginx.conf

# 1. Start FastAPI REST backend on 127.0.0.1:8000
echo "[1/3] Launching FastAPI REST API on 127.0.0.1:8000..."
uvicorn app.backend:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info &
BACKEND_PID=$!

# 2. Start Streamlit Web frontend on 127.0.0.1:8501
echo "[2/3] Launching Streamlit Web UI on 127.0.0.1:8501..."
streamlit run app/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

# Wait for backend health
echo "[3/3] Waiting for backend readiness probe..."
for i in {1..30}; do
    if curl -s -f http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo ">>> Backend is HEALTHY and ready to process inferences! <<<"
        break
    fi
    sleep 1
done

# Signal trap for clean termination
trap 'echo "Shutting down services..."; kill -TERM $BACKEND_PID $FRONTEND_PID $NGINX_PID 2>/dev/null; exit 0' SIGTERM SIGINT

echo ">>> Launching Nginx reverse proxy on 0.0.0.0:${PORT} <<<"
nginx -c /tmp/nginx.conf -g 'daemon off;' &
NGINX_PID=$!

# Wait for any process exit
wait -n $BACKEND_PID $FRONTEND_PID $NGINX_PID
