from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    FAKE = auto()
    GROQ = auto()
    HUGGINGFACE = auto()
    OLLAMA = auto()
    OPENAI = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "groq-llama-3.1-8b"
    LLAMA_33_70B = "groq-llama-3.3-70b"

    LLAMA_GUARD_3_8B = "groq-llama-guard-3-8b"


class HuggingFaceModelName(StrEnum):
    """https://huggingface.co/models?inference=warm"""

    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_V3 = "deepseek-v3"


class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | GroqModelName
    | HuggingFaceModelName
    | OllamaModelName
    | FakeModelName
)
