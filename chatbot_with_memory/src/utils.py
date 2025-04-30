from dataclasses import dataclass

@dataclass
class ChatbotConfig:
    """Configuration for the chatbot."""

    google_api_key: str
    model_name: str = 'gemini-2.0-flash'
    temperature: float = 0.7
    embedding_model: str = 'models/gemini-embedding-exp-03-07'
    chroma_persist_dir: str = './chroma_langchain_db'
    memory_k: int = 3 # Number of relevant memories to retrieve