# Multi-stage Dockerfile for PRISMX API Gateway and Retrieval Engine
# Stage 1: Dependency Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Runtime Image
FROM python:3.11-slim AS runtime

# Install curl for healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy project code and configuration
COPY src/ /app/src/
COPY config/ /app/config/
COPY CONFIG.yaml /app/CONFIG.yaml
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/pyproject.toml

# Create data directories with appropriate permissions
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

EXPOSE 8000

# Container healthcheck using GET /health
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "scripts/run_server.py"]
