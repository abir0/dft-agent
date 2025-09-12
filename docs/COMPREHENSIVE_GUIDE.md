# DFT Agent - Comprehensive Guide

## 🚀 Overview

The DFT Agent is an intelligent, tool-rich agent system for autonomous materials research and Density Functional Theory (DFT) workflows. Built with LangGraph (graph-based agent orchestration) and FastAPI, it features persistent conversation state, structured workspaces, and an extensible tool registry for real computational tasks.

## 📁 Project Structure

```
dft-agent/
├── backend/                    # Core backend services
│   ├── agents/                 # Agent implementations
│   │   ├── library/           # Agent library (chatbot + dft_agent)
│   │   │   ├── chatbot.py     # General-purpose chatbot agent
│   │   │   ├── dft_agent/     # DFT agent implementation
│   │   │   └── slurm_scheduler/ # SLURM job scheduler agent
│   │   ├── dft_tools/         # DFT tool implementations
│   │   ├── agent_manager.py   # Agent registry and management
│   │   ├── client.py          # Agent client interface
│   │   ├── llm.py            # LLM model configurations
│   │   └── tools.py          # General-purpose tools
│   ├── api/                   # FastAPI application
│   │   ├── endpoints/         # API route handlers
│   │   ├── main.py           # FastAPI app initialization
│   │   └── dependencies.py   # Dependency injection
│   ├── core/                  # Core data models and schemas
│   ├── utils/                 # Utility modules
│   └── web_browser/          # Web browsing capabilities
├── frontend/                  # Streamlit web interface
├── data/                     # Data storage and examples
│   ├── inputs/               # Input data (pseudopotentials, etc.)
│   ├── outputs/              # Calculation outputs
│   ├── raw_data/             # Raw datasets
│   └── examples/             # Example structures
├── docs/                     # Documentation
│   ├── guides/               # User guides
│   ├── api/                  # API documentation
│   └── examples/             # Code examples
├── scripts/                  # Deployment and utility scripts
└── tests/                    # Test suite
```

## 🛠 Technology Stack

### Core Framework
- **LangGraph**: Agent orchestration and state management
- **FastAPI**: High-performance API framework
- **Pydantic**: Data validation and serialization
- **SQLite**: Conversation checkpointing and persistence

### LLM Integrations
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-5
- **Groq**: Llama 3.1 8B, Llama 3.3 70B
- **HuggingFace**: DeepSeek R1, DeepSeek V3
- **Ollama**: Local model support
- **Fake**: Testing model

### Materials Science & DFT Libraries
- **ASE**: Atomic structure manipulation / I/O
- **Pymatgen**: Structure analysis & materials data
- **Materials Project API**: Remote materials database search
- **SeekPath**: Brillouin zone path generation
- **Quantum ESPRESSO**: DFT calculations

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd dft-agent
   uv sync  # or pip install -e .
   ```

2. **Environment setup:**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Start services:**
   ```bash
   ./start_services.sh
   ```

### Access Points
- **Web Interface**: http://localhost:8501
- **API Endpoint**: http://localhost:8083
- **API Documentation**: http://localhost:8083/docs

## 🤖 Available Agents

### 1. Chatbot Agent (Default)
General multi-domain assistant with:
- Web search (DuckDuckGo)
- Calculator & Python REPL
- Literature retrieval
- Asta MCP scholarly tools (when available)

### 2. DFT Agent
Expert computational materials assistant:
- Structure generation (bulk, supercell, slab, adsorption)
- Quantum ESPRESSO input generation and job submission
- Convergence testing (k-point, cutoff, slab thickness, vacuum)
- Materials Project integration
- Database utilities

### 3. SLURM Scheduler Agent
HPC job management and automation:
- Job submission and monitoring
- Queue management
- Resource allocation
- Job chaining and dependencies

## 🔧 Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| OPENAI_API_KEY | Enable OpenAI models |
| GROQ_API_KEY | Enable Groq Llama models |
| HF_API_KEY | Enable HuggingFace models |
| OLLAMA_MODEL / OLLAMA_BASE_URL | Local Ollama model config |
| AUTH_SECRET | Optional bearer token for API protection |
| MP_API_KEY | Materials Project queries |
| ASTA_KEY | Scholarly (Asta MCP) tools |
| DEFAULT_MODEL | Override initial default model |
| MODE=dev | Enables auto-reload & dev behaviors |

### Model Configuration
Models are enumerated in `backend/core/models.py` and configured in `backend/agents/llm.py`. The API surfaces availability via `/info`.

## 📊 DFT Workflow Examples

### Basic Structure Generation
```python
# Generate bulk structure
"Generate fcc Pt bulk structure"

# Create slab
"Create a (1 1 1) slab with 4 layers"

# Add adsorbate
"Add CO adsorbate to the surface"
```

### Quantum ESPRESSO Calculations
```python
# Generate QE input
"Make an scf QE input with 50 Ry cutoff"

# Submit job
"Submit local QE job"

# Check status
"Check job status and extract energy"
```

### Convergence Testing
```python
# K-point convergence
"Run k-point convergence test"

# Cutoff convergence
"Test cutoff energy convergence"

# Slab thickness
"Test slab thickness convergence"
```

## 🗂 Workspace Organization

Per-thread directories under `WORKSPACE/<thread_id>/` with standardized subfolders:

```
calculations/     # QE inputs and outputs
convergence_tests/ # Convergence test results
databases/        # Local calculation databases
kpaths/          # Brillouin zone paths
kpoints/         # K-point grids
optimized/       # Optimized structures
relaxed/         # Relaxed structures
results/         # Analysis results
structures/      # Generated structures
```

## 🔌 API Usage

### Core Endpoints
- `POST /agent/invoke` – Single-turn execution
- `POST /agent/stream` – Server-Sent Events streaming
- `POST /agent/feedback` – Record LangSmith feedback
- `POST /agent/history` – Retrieve conversation history
- `GET /health` – Service health check
- `GET /info` – Models and agents metadata

### Example API Call
```python
import httpx
import uuid

BASE = "http://localhost:8083"
thread_id = str(uuid.uuid4())
payload = {
    "message": "Generate fcc Pt bulk structure",
    "model": "gpt-4o-mini",
    "thread_id": thread_id,
}
response = httpx.post(f"{BASE}/agent/invoke", json=payload)
print(response.json())
```

### Streaming Example
```python
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", "http://localhost:8083/agent/stream", json={
            "message": "Create plan for adsorption energy of CO on Pt(111)",
            "model": "gpt-5",
            "thread_id": "example-thread-1"
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    print(line[6:])

asyncio.run(main())
```

## 🧪 Testing

Run the test suite:
```bash
# Basic tests
python -m pytest tests/

# Specific test categories
python -m pytest tests/test_dft_tools_basic.py
python -m pytest tests/test_dft_tools_comprehensive.py
```

## 🐳 Docker Support

Build and run with Docker:
```bash
docker-compose up --build
```

## 📈 Development Roadmap

### Phase 1 (Complete)
- ✅ LangGraph agent graphs, multi-model support, API, Streamlit
- ✅ Base tools and DFT workflow automation

### Phase 2 (Complete)
- ✅ Structure generation & manipulation
- ✅ Materials Project integration
- ✅ QE input + submission + parsing
- ✅ Convergence tests
- ✅ Local calculation database tooling
- ✅ Planning system

### Phase 3 (Planned)
- ⏳ Additional engines (VASP, Gaussian interfaces)
- ⏳ Advanced multi-branch workflows & dependency tracking
- ⏳ Automated job chaining & error recovery loops
- ⏳ Enhanced visualization (band structures, charge densities)
- ⏳ HPC / remote scheduler integration
- ⏳ Result provenance & reproducibility metadata

### Phase 4 (Stretch)
- ⏳ Active learning loops
- ⏳ Uncertainty-aware workflow steering
- ⏳ Multi-agent collaboration

## 🛡 Safety Notes

- The `python_repl` tool executes arbitrary code – restrict or disable in production
- All file operations are scoped to the workspace root
- API authentication is optional but recommended for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the excellent [LangGraph](https://github.com/langchain-ai/langgraph) framework
- Utilizes the [Materials Project](https://materialsproject.org/) ecosystem
- Inspired by the computational materials science community

---

**Happy computing! 🧪⚛️**
