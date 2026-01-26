# Open WebUI with MICA Pipe
# This Dockerfile extends Open WebUI with MICA integration

FROM ghcr.io/open-webui/open-webui:main

# Copy MICA pipe
COPY pipes/ /app/backend/pipelines/mica/

# Set environment variables for MICA integration
ENV MICA_BACKEND_URL=http://mica-backend:8000
ENV ENABLE_PIPELINES=true
