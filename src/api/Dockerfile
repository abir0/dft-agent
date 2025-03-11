FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

# Copy and install dependencies
COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project --no-dev

# Copy source files
COPY src/ .

CMD ["python", "run_service.py"]
