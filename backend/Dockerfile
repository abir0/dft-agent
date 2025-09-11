FROM python:3.12.3-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
	UV_PROJECT_ENVIRONMENT="/usr/local/" \
	UV_COMPILE_BYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# System build/runtime deps (minimal) + Quantum ESPRESSO (pw.x etc.)
# NOTE: The quantum-espresso package is available in Debian repos. If this ever fails
# you can replace with manual download of binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gfortran \
	wget \
	ca-certificates \
	libopenblas0 \
	liblapack3 \
	libfftw3-single3 \
	libopenmpi3 \
	openmpi-bin \
	quantum-espresso \
	&& rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
	uv sync --frozen --no-install-project --no-dev

# Copy backend source
COPY backend/ ./

# Expose QE bin dir (pw.x, etc.)
ENV QE_BIN_DIR="/usr/bin"
ENV PATH="$QE_BIN_DIR:$PATH"

# Workspace directory (mounted via docker-compose).
RUN mkdir -p /app/WORKSPACE
VOLUME ["/app/WORKSPACE"]

CMD ["python", "run_service.py"]
