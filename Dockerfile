# Production Dockerfile for IBM AI Enterprise Workflow Capstone
FROM python:3.11-slim-bookworm

# Set metadata
LABEL maintainer="AI Enterprise Team"
LABEL description="AI Enterprise Revenue Prediction & Monitoring API"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, data, models, tests, and notebooks
COPY . .

# Ensure working directory structure and permissions
RUN mkdir -p logs models data/cs-train data/cs-production reports/figures

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Expose microservice REST port
EXPOSE 8080

# Container healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Launch the production service
CMD ["python", "app.py"]
