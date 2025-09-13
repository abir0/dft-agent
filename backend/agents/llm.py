from functools import cache
from typing import TypeAlias

from langchain_cerebras import ChatCerebras
from langchain_community.chat_models import FakeListChatModel
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from backend.core import (
    AllModelEnum,
    CerebrasModelName,
    FakeModelName,
    GroqModelName,
    HuggingFaceModelName,
    OllamaModelName,
    OpenAIModelName,
)
from backend.settings import settings

_MODEL_TABLE = {
    FakeModelName.FAKE: "fake",
    OpenAIModelName.GPT_4O: "gpt-4o",
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_5: "gpt-5",
    OpenAIModelName.GPT_5_MINI: "gpt-5-mini",
    CerebrasModelName.GPT_OSS_120B: "gpt-oss-120b",
    CerebrasModelName.QWEN_235B_INSTRUCT: "qwen-3-235b-a22b-instruct-2507",
    CerebrasModelName.QWEN_235B_THINKING: "qwen-3-235b-a22b-thinking-2507",
    CerebrasModelName.LLAMA_4_MAVERICK: "llama-4-maverick-17b-128e-instruct",
    CerebrasModelName.LLAMA_33_70B: "llama-3.3-70b",
    OllamaModelName.OLLAMA_GENERIC: "ollama",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_GUARD_3_8B: "llama-guard-3-8b",
    HuggingFaceModelName.DEEPSEEK_R1: "deepseek-ai/DeepSeek-R1",
    HuggingFaceModelName.DEEPSEEK_V3: "deepseek-ai/DeepSeek-V3",
}

ModelT: TypeAlias = (
    ChatOpenAI | AzureChatOpenAI | ChatGroq | ChatOllama | ChatHuggingFace | ChatCerebras
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        # GPT-5 models only support default temperature (1.0)
        temp = (
            1.0
            if model_name in (OpenAIModelName.GPT_5, OpenAIModelName.GPT_5_MINI)
            else 0.3
        )
        return ChatOpenAI(
            model=api_model_name,
            temperature=temp,
            streaming=True,
            api_key=settings.OPENAI_API_KEY.get_secret_value()
            if settings.OPENAI_API_KEY
            else None,
        )
    if model_name in CerebrasModelName:
        return ChatCerebras(
            model=api_model_name,
            temperature=0.3,
            streaming=True,
            api_key=settings.CEREBRAS_API_KEY.get_secret_value()
            if settings.CEREBRAS_API_KEY
            else None,
        )
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0)
        return ChatGroq(model=api_model_name, temperature=0.5)
    if model_name in HuggingFaceModelName:
        llm = HuggingFaceEndpoint(
            repo_id=api_model_name,
            task="text-generation",
        )
        return ChatHuggingFace(llm=llm, temperature=0.5)
    if model_name in OllamaModelName:
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL,
                temperature=0.5,
                base_url=settings.OLLAMA_BASE_URL,
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
    if model_name in FakeModelName:
        return FakeListChatModel(
            responses=["This is a test response from the fake model."]
        )
