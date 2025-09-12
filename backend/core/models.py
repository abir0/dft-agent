from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    CEREBRAS = auto()
    FAKE = auto()
    GROQ = auto()
    HUGGINGFACE = auto()
    OLLAMA = auto()
    OPENAI = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"


class CerebrasModelName(StrEnum):
    """https://cloud.cerebras.ai/models"""

    GPT_OSS_120B = "gpt-oss-120b"
    QWEN_235B_INSTRUCT = "qwen-3-235b-a22b-instruct-2507"
    QWEN_235B_THINKING = "qwen-3-235b-a22b-thinking-2507"
    LLAMA_4_MAVERICK = "llama-4-maverick-17b-128e-instruct"
    LLAMA_33_70B = "llama-3.3-70b"


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
    | CerebrasModelName
    | GroqModelName
    | HuggingFaceModelName
    | OllamaModelName
    | FakeModelName
)
