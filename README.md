# SurfScriptors - LLM-powered DFT Agent Workflow Automation for Surface Science

An intelligent LLM-powered agent framework for autonomous materials research and DFT (Density Functional Theory) calculations, built with LangGraph and FastAPI.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync --dev
# or: pip install -e .

# Configure environment
cp env.example .env
# Edit .env with your API keys (at least one required):
# OPENAI_API_KEY="sk-..."
# GROQ_API_KEY="gsk_..."
# HF_API_KEY="hf_..."

# Start services
./scripts/run.sh

# Access interface
open http://localhost:8501
```

## 📚 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [DFT Tools Guide](#dft-tools-guide)
- [Testing](#testing)
- [API Usage](#api-usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## 🛠️ Installation

### Prerequisites
- **Python**: 3.12+ (required)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space

### Method 1: Using uv (Recommended)
```bash
git clone https://github.com/your-username/dft-agent.git
cd dft-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Method 2: Using pip
```bash
git clone https://github.com/your-username/dft-agent.git
cd dft-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Method 3: Using Docker
```bash
git clone https://github.com/your-username/dft-agent.git
cd dft-agent
docker-compose up --build
```

## ⚙️ Configuration

### Environment Setup
```bash
cp env.example .env
```

### Required API Keys (at least one)
```bash
# LLM API Keys
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

### DFT Software Setup (Optional)
```bash
# Quantum ESPRESSO
sudo apt-get install quantum-espresso  # Ubuntu/Debian
brew install quantum-espresso          # macOS
export QE_BIN="/path/to/pw.x"

# VASP (if available)
export VASP_BIN="/path/to/vasp_std"
```

## ⚛️ DFT Tools Guide

The DFT Agent provides **29 specialized tools** across **8 categories** for comprehensive materials research.

### Tool Categories

#### 1. Structure Generation & Manipulation (5 tools)
- **`generate_bulk`** - Create bulk crystal structures
- **`create_supercell`** - Generate supercells
- **`generate_slab`** - Create surface slabs
- **`add_adsorbate`** - Add adsorbates to surfaces
- **`add_vacuum`** - Add vacuum layers

#### 2. DFT Calculation & Optimization (4 tools)
- **`run_dft_calculation`** - Execute DFT calculations
- **`optimize_structure_dft`** - Optimize structures
- **`relax_slab_dft`** - **NEW**: Relax slabs with layer fixing for catalysis
- **`test_hydrogen_atom`** - Simple test calculation

#### 3. Structure Optimization (2 tools)
- **`relax_bulk`** - Relax bulk structures
- **`relax_slab`** - Relax slab structures (ASE-based)

#### 4. K-point Analysis (2 tools)
- **`generate_kpoint_mesh`** - Generate k-point meshes
- **`get_kpath_bandstructure`** - Generate k-paths for band structure

#### 5. Materials Database (4 tools)
- **`search_materials_project`** - Search Materials Project
- **`analyze_crystal_structure`** - Analyze crystal properties
- **`find_pseudopotentials`** - Find pseudopotentials
- **`calculate_formation_energy`** - Calculate formation energies

#### 6. Quantum ESPRESSO Interface (5 tools)
- **`generate_qe_input`** - Generate QE input files
- **`submit_local_job`** - Submit local jobs
- **`check_job_status`** - Monitor job status
- **`read_output_file`** - Parse output files
- **`extract_energy`** - Extract energy values

#### 7. Convergence Testing (4 tools)
- **`kpoint_convergence_test`** - Test k-point convergence
- **`cutoff_convergence_test`** - Test cutoff convergence
- **`slab_thickness_convergence`** - Test slab thickness
- **`vacuum_convergence_test`** - Test vacuum thickness

#### 8. Database Management (7 tools)
- **`create_calculations_database`** - Create SQLite database
- **`store_calculation`** - Store calculation results
- **`update_calculation_status`** - Update calculation status
- **`store_adsorption_energy`** - Store adsorption data
- **`query_calculations`** - Query database
- **`export_results`** - Export results
- **`search_similar_calculations`** - Find similar calculations

### Usage Examples

#### Basic Workflow
```python
from backend.agents.dft_tools import (
    generate_bulk, generate_slab, relax_slab_dft, 
    add_adsorbate, run_dft_calculation
)

# 1. Generate bulk structure
bulk = generate_bulk.invoke({
    "element": "Pt",
    "crystal_structure": "fcc",
    "lattice_parameter": 3.92,
    "output_file": "Pt_bulk.xyz"
})

# 2. Create slab
slab = generate_slab.invoke({
    "structure_file": "Pt_bulk.xyz",
    "miller_indices": [1, 1, 1],
    "layers": 5,
    "vacuum": 15.0,
    "output_file": "Pt_slab.xyz"
})

# 3. Relax slab with layer fixing (for catalysis)
relaxed = relax_slab_dft.invoke({
    "structure_file": "Pt_slab.xyz",
    "output_dir": "Pt_relax",
    "fixed_layers": 2,  # Fix 2 bottom layers
    "ecutwfc": 40.0,
    "kpts": [8, 8, 1]
})

# 4. Add adsorbate
with_adsorbate = add_adsorbate.invoke({
    "structure_file": "Pt_relaxed.xyz",
    "adsorbate": "CO",
    "position": [0.0, 0.0, 2.5],
    "output_file": "Pt_CO.xyz"
})

# 5. Run DFT calculation
calculation = run_dft_calculation.invoke({
    "structure_file": "Pt_CO.xyz",
    "output_dir": "Pt_CO_calc",
    "ecutwfc": 40.0,
    "kpts": [8, 8, 1],
    "calculation_type": "scf"
})
```

#### Convergence Testing
```python
from backend.agents.dft_tools import (
    kpoint_convergence_test, cutoff_convergence_test
)

# Test k-point convergence
kpoint_results = kpoint_convergence_test.invoke({
    "structure_file": "Si_bulk.xyz",
    "output_dir": "kpoint_conv",
    "kpoint_range": [2, 4, 6, 8, 10],
    "ecutwfc": 30.0
})

# Test cutoff convergence
cutoff_results = cutoff_convergence_test.invoke({
    "structure_file": "Si_bulk.xyz",
    "output_dir": "cutoff_conv",
    "cutoff_range": [20, 30, 40, 50],
    "kpts": [6, 6, 6]
})
```

### Best Practices

#### Parameter Selection
- **Metals**: Higher cutoffs (35-50 Ry), Fermi-Dirac smearing
- **Semiconductors**: Moderate cutoffs (25-35 Ry), Gaussian smearing
- **Insulators**: Lower cutoffs (20-30 Ry), Gaussian smearing

#### K-point Mesh
- **Bulk**: Dense mesh (6x6x6 to 12x12x12)
- **Slabs**: Dense x-y, single z (8x8x1 to 12x12x1)
- **Molecules**: Single k-point (1x1x1)

## 🧪 Testing

### Basic Tests (Recommended First)
```bash
# Run basic functionality tests
python tests/test_dft_tools_basic.py
```

**Expected Output:**
```
🚀 Starting Basic DFT Tools Testing
==================================================
Total Tests: 29
✅ Passed: 28
❌ Failed: 1
📈 Success Rate: 96.6%
```

### Comprehensive Tests
```bash
# Run full DFT tool tests (requires DFT software)
python tests/test_dft_tools_comprehensive.py
```

**Expected Output:**
```
🚀 Starting Comprehensive DFT Tools Testing
============================================================
Total Tests: 36
✅ Passed: 32
❌ Failed: 4
📈 Success Rate: 88.9%
```

### Test Categories
- **Module imports** (8 tests)
- **Tool registry** (7 tests)
- **Structure operations** (2 tests)
- **Database operations** (2 tests)
- **Configuration validation** (3 tests)
- **File I/O operations** (3 tests)
- **Environment setup** (3 tests)

## 🌐 API Usage

### Web Interface
1. **Open**: http://localhost:8501
2. **Select Agent**: Choose "DFT Agent" from dropdown
3. **Ask Questions**: 
   - "Generate a platinum bulk structure"
   - "Create a (111) surface slab"
   - "Relax the slab with 2 fixed layers"
   - "Add a CO molecule to the surface"

### REST API
```bash
# Chat with agent
curl -X POST "http://localhost:8080/api/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Generate a silicon bulk structure",
    "agent_type": "dft_agent"
  }'

# Get available tools
curl -X GET "http://localhost:8080/api/agent/tools"
```

### Python Client
```python
import requests

response = requests.post("http://localhost:8080/api/agent/chat", json={
    "message": "Generate a platinum bulk structure",
    "agent_type": "dft_agent"
})

print(response.json())
```

## 🏗️ Project Structure

```
dft-agent/
├── README.md              # This comprehensive guide
├── pyproject.toml         # Project configuration
├── setup.py               # Package setup
├── env.example            # Environment template
├── docker-compose.yaml    # Docker configuration
├── backend/               # Core backend services
│   ├── agents/            # Agent implementations
│   │   ├── dft_tools/     # 29 DFT-specific tools
│   │   ├── library/       # Agent library
│   │   └── ...
│   ├── api/               # FastAPI application
│   ├── core/              # Core data models
│   └── settings.py        # Configuration management
├── frontend/              # Streamlit web interface
├── tests/                 # Test suites
├── scripts/               # Utility scripts
├── data/                  # Data storage
│   ├── inputs/            # Input data and pseudopotentials
│   └── outputs/           # Calculation results
├── structures/            # Structure files
├── logs/                  # Log files
└── WORKSPACE/             # User-specific workspaces
```

## 🔧 Key Features

- **29 DFT Tools** across 8 categories for materials research
- **LLM Integration** with GPT-5, Groq, Ollama, and HuggingFace
- **Surface Science** tools for heterogeneous catalysis studies
- **REST API** and Web Interface for easy access
- **Docker Support** for containerized deployment
- **Database Management** with SQLite-based calculation tracking
- **Convergence Testing** for automated parameter optimization

## 🆘 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'backend'
# Solution:
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

#### Missing Dependencies
```bash
# Error: ImportError: No module named 'ase'
# Solution:
uv sync --dev
# or
pip install -e .
```

#### DFT Software Not Found
```bash
# Error: Executable 'pw.x' not found in PATH
# Solution:
sudo apt-get install quantum-espresso
# Or set environment variable
export QE_BIN="/path/to/pw.x"
```

#### API Key Issues
```bash
# Error: ValueError: At least one LLM API key must be provided
# Solution:
# Edit .env file with your API keys
export OPENAI_API_KEY="sk-..."
export MP_API_KEY="your_mp_key"
```

#### Permission Errors
```bash
# Error: PermissionError: [Errno 13] Permission denied
# Solution:
chmod +x scripts/*.sh
chmod +x tests/*.py
```

### Getting Help
1. **Check Logs**: `tail -f logs/service.log`
2. **Run Tests**: `python tests/test_dft_tools_basic.py`
3. **Verify Setup**: Check environment variables and API keys
4. **Create Issue**: If problems persist, create an issue in the repository

## 📊 System Status

- **Total Tools**: 29 DFT tools across 8 categories
- **Test Coverage**: 96.6% success rate on basic functionality
- **Documentation**: Comprehensive single-file guide
- **API Support**: RESTful API with OpenAPI documentation
- **Status**: Production Ready

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Use Cases

### Materials Research
- Structure generation and optimization
- Surface science and catalysis studies
- Convergence testing and parameter optimization
- Materials database integration

### DFT Calculations
- Quantum ESPRESSO and VASP integration
- Automated workflow management
- Database tracking and result storage
- Web-based interface for researchers

---

**Ready to start?** Run `./scripts/run.sh` and open http://localhost:8501!