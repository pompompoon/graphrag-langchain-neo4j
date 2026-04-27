"""
LangChain統合設定
- LLM/Embeddings: PROVIDER で ollama / gemini を切り替え
- ストレージ: STORAGE で file / neo4j を切り替え
"""

import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

load_dotenv()

# ============================================================
# 設定値（.env から読み込み）
# ============================================================

# LLMプロバイダ
PROVIDER = os.getenv("PROVIDER", "ollama")

# ストレージ
STORAGE = os.getenv("STORAGE", "file")  # "file" or "neo4j"

# Ollama設定
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EXTRACT_MODEL = os.getenv("OLLAMA_EXTRACT_MODEL", "qwen2.5:7b")

# Gemini設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EXTRACT_MODEL = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

# Neo4j設定
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def get_chat_llm(temperature: float = 0.3) -> BaseChatModel:
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL, google_api_key=GEMINI_API_KEY,
            temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL,
            temperature=temperature)


def get_extract_llm(temperature: float = 0.0) -> BaseChatModel:
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_EXTRACT_MODEL, google_api_key=GEMINI_API_KEY,
            temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_EXTRACT_MODEL, base_url=OLLAMA_BASE_URL,
            temperature=temperature)


def get_embeddings() -> Embeddings:
    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL, google_api_key=GEMINI_API_KEY)
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_embed_dim() -> int:
    return 768


def get_provider_info() -> dict:
    info = {"storage": STORAGE}
    if PROVIDER == "gemini":
        info.update({"provider": "gemini", "chat_model": GEMINI_CHAT_MODEL,
                     "embed_model": GEMINI_EMBED_MODEL})
    else:
        info.update({"provider": "ollama", "chat_model": OLLAMA_CHAT_MODEL,
                     "embed_model": OLLAMA_EMBED_MODEL})
    return info
