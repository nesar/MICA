# MICA Deployment Guide

This guide covers deployment options for MICA, from local development to production cloud hosting.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring and Logging](#monitoring-and-logging)

## Local Development

### Prerequisites

- Python 3.10 or higher
- pip or conda for package management
- Access to LLM backend (Argo VPN or Google API key)

### Setup

1. **Create virtual environment:**
   ```bash
   cd /path/to/MICA/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   # Copy example configuration
   cp ../.env.example ../.env

   # Edit .env with your settings
   # For Argo (requires VPN):
   export MICA_LLM_PROVIDER=argo
   export ARGO_USERNAME=your_username

   # For Gemini (public):
   export MICA_LLM_PROVIDER=gemini
   export GOOGLE_API_KEY=your_api_key
   ```

4. **Run the backend:**
   ```bash
   python -m mica.api.main
   ```

5. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

### Development with Hot Reload

```bash
uvicorn mica.api.main:app --reload --port 8000
```

## Docker Deployment

### Basic Docker Compose

1. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Build and start:**
   ```bash
   docker-compose up -d --build
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f mica-backend
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

### Services

| Service | Port | Description |
|---------|------|-------------|
| mica-backend | 8000 | FastAPI backend API |
| open-webui | 3000 | Open WebUI interface |

### Volumes

| Volume | Purpose |
|--------|---------|
| mica-data | General data storage |
| open-webui-data | Open WebUI persistence |
| ./backend/sessions | Session logs (mounted) |
| ./backend/chroma_db | Vector database (mounted) |

## Cloud Deployment

### Requirements for Production

1. **Reverse Proxy (nginx)**: SSL termination, load balancing
2. **Domain Name**: For HTTPS access
3. **SSL Certificate**: Let's Encrypt or commercial
4. **Persistent Storage**: For sessions and vector database

### Docker Compose for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  mica-backend:
    image: mica-backend:latest
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
    environment:
      - MICA_LOG_LEVEL=WARNING
      - MICA_CORS_ORIGINS=https://yourdomain.com
    volumes:
      - mica-sessions:/app/sessions
      - mica-chroma:/app/chroma_db
    networks:
      - mica-internal

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    environment:
      - WEBUI_AUTH=true
      - ENABLE_SIGNUP=false
    volumes:
      - webui-data:/app/backend/data
    networks:
      - mica-internal

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - mica-backend
      - open-webui
    networks:
      - mica-internal
      - mica-external

volumes:
  mica-sessions:
  mica-chroma:
  webui-data:

networks:
  mica-internal:
    internal: true
  mica-external:
```

### Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream mica_backend {
        server mica-backend:8000;
    }

    upstream open_webui {
        server open-webui:8080;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;

        # Open WebUI
        location / {
            proxy_pass http://open_webui;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # MICA API
        location /api/ {
            proxy_pass http://mica_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # WebSocket
        location /ws/ {
            proxy_pass http://mica_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### SSL with Let's Encrypt

```bash
# Install certbot
apt-get install certbot

# Get certificate
certbot certonly --standalone -d yourdomain.com

# Copy certificates
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./certs/
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./certs/

# Set up auto-renewal
certbot renew --dry-run
```

## Production Considerations

### Security

1. **Environment Variables**: Never commit secrets to git
   ```bash
   # Use .env files or secrets management
   docker secret create argo_username ./secrets/argo_username
   ```

2. **Authentication**: Enable Open WebUI authentication
   ```yaml
   environment:
     - WEBUI_AUTH=true
     - ENABLE_SIGNUP=false
   ```

3. **API Security**: Configure API key authentication
   ```bash
   export MICA_API_KEY=$(openssl rand -hex 32)
   ```

4. **Network Isolation**: Use internal Docker networks

### Scaling

1. **Horizontal Scaling**: Multiple backend replicas
   ```yaml
   deploy:
     replicas: 3
   ```

2. **Load Balancing**: Configure nginx upstream
   ```nginx
   upstream mica_backend {
       least_conn;
       server mica-backend-1:8000;
       server mica-backend-2:8000;
       server mica-backend-3:8000;
   }
   ```

3. **Shared Storage**: Use network-attached storage for sessions
   ```yaml
   volumes:
     mica-sessions:
       driver: local
       driver_opts:
         type: nfs
         o: addr=nfs-server,rw
         device: ":/exports/mica-sessions"
   ```

### Backup

1. **Session Data**:
   ```bash
   # Backup sessions
   tar -czf sessions-backup-$(date +%Y%m%d).tar.gz ./backend/sessions/
   ```

2. **Vector Database**:
   ```bash
   # Backup ChromaDB
   tar -czf chroma-backup-$(date +%Y%m%d).tar.gz ./backend/chroma_db/
   ```

3. **Automated Backups**:
   ```bash
   # Add to crontab
   0 2 * * * /path/to/backup-script.sh
   ```

## Monitoring and Logging

### Health Checks

The backend provides a health endpoint:
```bash
curl http://localhost:8000/health
```

### Logging

1. **Application Logs**:
   ```bash
   docker-compose logs -f mica-backend
   ```

2. **Log Aggregation**: Configure log forwarding
   ```yaml
   services:
     mica-backend:
       logging:
         driver: "json-file"
         options:
           max-size: "10m"
           max-file: "3"
   ```

3. **Session Audit Logs**: Located in `/sessions/{session_id}/`
   - `metadata.json`: Session metadata
   - `query.txt`: Original query
   - `plan.json`: Analysis plan
   - `agent_logs/`: Per-agent execution logs

### Metrics (Optional)

Add Prometheus metrics:
```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**:
   - Verify VPN connection for Argo
   - Check API key for Gemini
   - Review logs: `docker-compose logs mica-backend`

2. **WebSocket Disconnections**:
   - Check nginx timeout settings
   - Verify proxy headers

3. **Memory Issues**:
   - Increase container memory limits
   - Monitor with `docker stats`

### Debug Mode

Enable debug logging:
```bash
export MICA_LOG_LEVEL=DEBUG
```

### Support

For issues, please file a report at the project repository.
