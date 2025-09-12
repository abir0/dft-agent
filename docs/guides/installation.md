# Installation Guide

## Prerequisites

### System Requirements
- **Python**: 3.12+ (required)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space for dependencies and data

### Required Software
- **Git**: For cloning the repository
- **uv** (recommended) or **pip**: Package management
- **Docker** (optional): For containerized deployment

### Optional Software
- **Quantum ESPRESSO**: For DFT calculations
- **VASP**: For advanced DFT calculations
- **ASE**: Atomic Simulation Environment
- **Pymatgen**: Materials analysis

## Installation Methods

### Method 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/dft-agent.git
cd dft-agent

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate

# Add project to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/dft-agent.git
cd dft-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Add project to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Method 3: Using Docker

```bash
# Clone the repository
git clone https://github.com/your-username/dft-agent.git
cd dft-agent

# Build and run with Docker Compose
docker-compose up --build
```

## Configuration

### 1. Environment Setup

Copy the example environment file:
```bash
cp env.example .env
```

### 2. API Keys Configuration

Edit `.env` file with your API keys:

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY="sk-..."
GROQ_API_KEY="gsk_..."
HF_API_KEY="hf_..."

# Materials Project API Key (optional)
MP_API_KEY="your_materials_project_key"

# LangSmith Settings (optional)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="dft-agent"
```

### 3. DFT Software Setup (Optional)

#### Quantum ESPRESSO
```bash
# Install Quantum ESPRESSO
# Ubuntu/Debian
sudo apt-get install quantum-espresso

# macOS with Homebrew
brew install quantum-espresso

# Set environment variable
export QE_BIN="/path/to/pw.x"
```

#### VASP (if available)
```bash
# Set environment variable
export VASP_BIN="/path/to/vasp_std"
```

## Verification

### 1. Basic Installation Test
```bash
# Run basic structure test
python tests/test_basic_structure.py
```

### 2. Full Functionality Test
```bash
# Run comprehensive tests
python tests/test_dft_tools_basic.py
```

### 3. Start Services
```bash
# Start backend service
./scripts/run.sh

# Or manually:
# Backend: python backend/run_service.py
# Frontend: streamlit run frontend/app.py
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure Python path is set
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

#### Missing Dependencies
```bash
# Reinstall dependencies
uv sync --dev
# or
pip install -e .
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

#### DFT Software Not Found
- Check if Quantum ESPRESSO is installed
- Verify PATH environment variable
- Set QE_BIN environment variable

## Next Steps

1. **Configure API Keys**: Edit `.env` file
2. **Run Tests**: Verify installation
3. **Start Services**: Use `./scripts/run.sh`
4. **Access Interface**: Open http://localhost:8501

## Support

- **Documentation**: See [Quick Start Guide](quickstart.md)
- **Issues**: Check [Troubleshooting Guide](troubleshooting.md)
- **Questions**: Create an issue in the repository
