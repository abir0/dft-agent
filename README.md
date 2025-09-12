# DFT Agent - LLM-powered Workflow Automation for Materials Science

An intelligent, tool-rich agent system for autonomous materials research and Density Functional Theory (DFT) workflows. Built with LangGraph and FastAPI, featuring persistent conversation state, structured workspaces, and an extensible tool registry for real computational tasks.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd dft-agent
uv sync

# Configure environment
cp env.example .env
# Edit .env with your API keys

# Start services
./start_services.sh
```

**Access Points:**
- 🌐 **Web Interface**: http://localhost:8501
- 🔧 **API Endpoint**: http://localhost:8083
- 📚 **API Docs**: http://localhost:8083/docs

## ✨ Key Features

- **Multi-Agent System**: Chatbot, DFT Agent, and SLURM Scheduler
- **DFT Workflows**: Structure generation, QE calculations, convergence testing
- **Materials Integration**: Materials Project API, structure analysis
- **HPC Support**: SLURM job management and automation
- **Persistent State**: Thread-based conversation continuity
- **Streaming API**: Real-time response streaming
- **Web Interface**: User-friendly Streamlit frontend

## 📚 Documentation

- **[Comprehensive Guide](docs/COMPREHENSIVE_GUIDE.md)** - Complete documentation
- **[Installation Guide](docs/guides/installation.md)** - Setup instructions
- **[SLURM Integration](docs/guides/SLURM_SCHEDULER.md)** - HPC job management
- **[Testing Guide](docs/guides/testing.md)** - Test suite documentation
- **[API Documentation](http://localhost:8083/docs)** - Interactive API docs

## 🤖 Available Agents

### Chatbot Agent (Default)
General multi-domain assistant with web search, calculator, Python REPL, and literature tools.

### DFT Agent
Expert computational materials assistant for:
- Structure generation (bulk, supercell, slab, adsorption)
- Quantum ESPRESSO input generation and job submission
- Convergence testing (k-point, cutoff, slab thickness, vacuum)
- Materials Project integration
- Database utilities

### SLURM Scheduler Agent
HPC job management and automation for:
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

### Docker Support

```bash
docker-compose up --build
```

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/

# Specific test categories
python -m pytest tests/test_dft_tools_basic.py
python -m pytest tests/test_dft_tools_comprehensive.py
```

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

### 🔐 Environment Variables (Summary)

| Variable | Purpose |
|----------|---------|
| OPENAI_API_KEY | Enable OpenAI models |
| GROQ_API_KEY | Enable Groq Llama models |
| HF_API_KEY | Enable HuggingFace endpoint models |
| OLLAMA_MODEL / OLLAMA_BASE_URL | Local Ollama model config |
| AUTH_SECRET | Optional bearer token for API protection |
| MP_API_KEY | Materials Project queries |
| ASTA_KEY | Scholarly (Asta MCP) tools |
| DEFAULT_MODEL | Override initial default model |
| MODE=dev | Enables auto-reload & dev behaviors |

At least one model provider key (or USE_FAKE_MODEL=true) must be set; otherwise initialization will fail.

### 🗂 Workspace Layout

Per-thread directories under `WORKSPACE/<thread_id>/` with standardized subfolders:

```text
calculations/  convergence_tests/  databases/  kpaths/  kpoints/  optimized/  relaxed/  results/  structures/
```

Tools automatically place generated artifacts in the correct subfolder (e.g. QE inputs under `calculations/`). Return payloads are concise; large artifacts are written to disk for persistence.

### 🛡 Safety Notes

- The `python_repl` tool executes arbitrary code – restrict or disable in production deployments.
- All file operations are scoped to the workspace root; avoid passing untrusted absolute paths.

### ❓ Support

Open an issue or discussion for feature requests, bug reports, or extension ideas.

**Happy computing! 🧪⚛️**
