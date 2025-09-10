# DFT Agent

An intelligent LLM-powered agent framework for autonomous materials research and DFT (Density Functional Theory) calculations, built with LangGraph and FastAPI.

## 🚀 Project Status

This is the foundational codebase for a DFT agent system. Currently implemented:

- ✅ Core agentic framework using LangGraph
- ✅ RESTful API with FastAPI
- ✅ Multi-model LLM support (OpenAI, Groq, Ollama, HuggingFace)
- ✅ Basic chatbot agent with web search, calculator, and Python REPL tools
- ✅ Streamlit frontend interface
- ⏳ DFT-specific tools and calculations (planned)
- ⏳ Materials database integration (planned)

## 📁 Project Structure

```text
dft-agent/
├── backend/                    # Core backend services
│   ├── agents/                 # Agent implementations
│   │   ├── library/           # Agent library
│   │   │   ├── chatbot.py     # Basic chatbot agent
│   │   │   └── dft_agent/     # DFT agent (placeholder)
│   │   ├── dft_tools/         # DFT-specific tools (empty)
│   │   ├── agent_manager.py   # Agent registry and management
│   │   ├── client.py          # Agent client interface
│   │   ├── llm.py            # LLM model configurations
│   │   └── tools.py          # General-purpose tools
│   ├── api/                   # FastAPI application
│   │   ├── endpoints/         # API route handlers
│   │   ├── main.py           # FastAPI app initialization
│   │   └── dependencies.py   # Dependency injection
│   ├── core/                  # Core data models and schemas
│   │   ├── models.py         # Pydantic models
│   │   ├── schema.py         # API schemas
│   │   └── exceptions.py     # Custom exceptions
│   ├── utils/                 # Utility modules
│   └── web_browser/          # Web browsing capabilities
├── frontend/                  # Streamlit web interface
│   └── app.py                # Main Streamlit application
├── data/                     # Data storage
│   └── raw_data/             # Raw datasets (adsorption data)
├── scripts/                  # Deployment and utility scripts
└── logs/                     # Application logs
```

## 🛠 Technology Stack

### Core Framework

- **LangGraph**: Agent orchestration and state management
- **FastAPI**: High-performance API framework
- **Pydantic**: Data validation and serialization
- **SQLite**: Conversation checkpointing and persistence

### LLM Integrations

- **OpenAI GPT Models**: GPT-4, GPT-3.5-turbo
- **Groq**: High-speed inference
- **Ollama**: Local model deployment
- **HuggingFace**: Open-source model access

### Materials Science Libraries

- **ASE (Atomic Simulation Environment)**: Atomic structure manipulation
- **Pymatgen**: Materials analysis and crystal structure tools
- **Materials Project API**: Materials database integration

### Tools & Utilities

- **DuckDuckGo Search**: Web search capabilities
- **Python REPL**: Code execution environment
- **Calculator**: Mathematical computations
- **Streamlit**: Interactive web interface

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd dft-agent
   ```

2. **Install dependencies:**

   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Environment setup:**

   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys
   # Required: OPENAI_API_KEY, GROQ_API_KEY, etc.
   ```

### Running the Application

#### Backend API Server

```bash
# Development mode
cd backend
python run_service.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Interface

```bash
# Streamlit app
cd frontend
streamlit run app.py
```

#### Using Docker

```bash
# Build and run with docker-compose
docker-compose up --build
```

### API Usage

The API provides RESTful endpoints for agent interaction:

```python
import httpx

# Chat with the agent
response = httpx.post("http://localhost:8000/chat", json={
    "message": "What is the band gap of silicon?",
    "model": "gpt-4",
    "agent": "chatbot"
})

# Stream responses
async with httpx.stream("POST", "http://localhost:8000/stream", json={
    "message": "Calculate the adsorption energy of CO on Pt(111)",
}) as response:
    async for chunk in response.aiter_text():
        print(chunk, end="")
```

## 🔧 Configuration

### Model Configuration

Edit `backend/core/models.py` to add new LLM providers or models:

```python
class OpenAIModelName(str, Enum):
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
```

### Agent Configuration

Register new agents in `backend/agents/agent_manager.py`:

```python
agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot", graph=chatbot),
    "dft_agent": Agent(description="DFT calculations agent", graph=dft_agent),
}
```

## 📊 Current Agents

### Chatbot Agent

- **Purpose**: General-purpose conversational agent
- **Tools**: Web search, calculator, Python REPL
- **Capabilities**:
  - Answer general questions
  - Perform calculations
  - Execute Python code
  - Search the web for current information

### DFT Agent (Planned)

- **Purpose**: Specialized for materials science and DFT calculations
- **Tools**: ASE, Pymatgen, quantum chemistry packages
- **Capabilities**:
  - Structure optimization
  - Electronic property calculations
  - Adsorption energy calculations
  - Band structure analysis

## 📈 Development Roadmap

### Phase 1: Foundation (✅ Complete)

- [x] Core agent framework with LangGraph
- [x] Multi-model LLM support
- [x] Basic API endpoints
- [x] Streamlit interface
- [x] Basic tools (search, calculator, REPL)

### Phase 2: DFT Integration (🚧 In Progress)

- [ ] ASE integration for structure manipulation
- [ ] Pymatgen tools for materials analysis
- [ ] Quantum ESPRESSO interface
- [ ] VASP interface
- [ ] Gaussian interface

### Phase 3: Advanced Features

- [ ] Workflow orchestration for multi-step calculations
- [ ] Results database and caching
- [ ] Visualization tools for structures and properties
- [ ] Batch job management
- [ ] Integration with HPC systems

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
- Inspired by the needs of the computational materials science community
