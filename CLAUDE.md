# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DFT-Agent is an LLM-powered agent system for materials research and DFT (Density Functional Theory) calculations. It uses LangGraph for agent orchestration and provides both API and Streamlit UI interfaces.

## Key Commands

### Development Setup
```bash
# Install dependencies using uv (preferred)
uv sync

# Or using pip
pip install -e .
```

### Running the Application
```bash
# Start both API service and Streamlit frontend
./scripts/run.sh

# Or run components individually:
# API Service
python backend/run_service.py

# Streamlit Frontend  
streamlit run frontend/app.py

# Stop all services
./scripts/stop.sh
```

### Code Quality
```bash
# Run linting with ruff
ruff check backend/
ruff format backend/

# Run tests (pytest is installed as dev dependency)
pytest
```

### LangGraph Development
```bash
# Deploy agent using LangGraph CLI
langgraph up

# Test agent locally
python backend/run_agent.py
```

## Architecture

### Backend Structure
- **backend/agents/**: LangGraph agent implementations
  - `agent_manager.py`: Registry of available agents
  - `library/chatbot.py`: Main chatbot agent graph
  - `tools.py`: Agent tools (calculator, python_repl)
  - `llm.py`: LLM provider configuration
  
- **backend/api/**: FastAPI application
  - REST endpoints for agent interaction
  - WebSocket support for streaming responses
  
- **backend/core/**: Core data models and types
  
- **backend/utils/**: Utility functions and helpers

### Frontend
- **frontend/app.py**: Streamlit application for user interface
- Connects to backend API for agent interactions

### Configuration
- Environment variables loaded from `.env` file
- Settings centralized in `backend/settings.py`
- Supports multiple LLM providers: OpenAI, Groq, HuggingFace, Ollama
- Materials Project API integration via MP_API_KEY

## LLM Configuration

The system supports multiple LLM providers configured via environment variables:
- `OPENAI_API_KEY`: For OpenAI models
- `GROQ_API_KEY`: For Groq models  
- `HF_API_KEY`: For HuggingFace models
- `OLLAMA_MODEL`: For local Ollama models
- `DEFAULT_MODEL`: Set the default model to use

## Development Notes

- Python 3.12 required (enforced in pyproject.toml)
- Uses `uv` for package management (see uv.lock)
- Ruff configured for linting with line length 90
- LangGraph agents defined in `langgraph.json`
- Docker Compose available for containerized deployment