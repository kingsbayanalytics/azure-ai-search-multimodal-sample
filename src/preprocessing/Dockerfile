# Cost-Effective Document Preprocessing Pipeline Container
# Multi-stage build for optimized production image

# Build stage
FROM python:3.12-slim as builder

# Install system dependencies required for document processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r preprocessing && useradd -r -g preprocessing preprocessing

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/temp /app/output && \
    chown -R preprocessing:preprocessing /app

# Switch to non-root user
USER preprocessing

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "from config.settings import get_settings; print('Health OK')" || exit 1

# Expose monitoring port
EXPOSE 8000

# Default command
CMD ["python", "document_processor.py", "--help"]

# Labels for metadata
LABEL maintainer="preprocessing-team" \
      version="1.0" \
      description="Cost-effective document preprocessing pipeline" \
      cost_savings="70-85% vs Azure Document Intelligence" 