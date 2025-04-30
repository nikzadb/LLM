import os
from typing import Dict, List, Any
import logging
from datetime import datetime

from langchain.memory import ConversationBufferMemory

from langchain_chroma import Chroma
from langchain.schema import Document

logger = logging.getLogger(__name__)

class ChatbotMemory:
    """Memory management for the chatbot."""

    def __init__(self, embeddings, vector_store: Chroma, memory_k: int = 3):
        """Initilise the memory system."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.memory_k = memory_k
        self.short_term_memory = ConversationBufferMemory()

        # Track conversation metadata
        self.conversation_count = 0 
        self.last_interaction_time = None

    def add_to_memory(self, human_input: str, ai_response: str) -> None:
        """Add conversation to both short-term and long-term memory"""

        # Update metadata
        self.conversation_count += 1
        self.last_interaction_time = datetime.now()
        metadata = {
            "timestamp": self.last_interaction_time.isoformat(),
            "conversation_id": self.conversation_count
        }        

        # Short-term memory
        self.short_term_memory.save_context(
            {"input": human_input},
            {"output": ai_response}
        )
        
        # long-term memory
        memory_text = f"Human: {human_input}\nAi: {ai_response}"

        self.vector_store.add_texts(
            texts = [memory_text],
            metadata = [metadata]
        )

    def get_relevant_memories(self, query: str) -> str:
        """Retrieve relevant past conversations."""

        docs = self.vector_store.similarity_search(query, k=self.memory_k)

        formated_memories = []
        for doc in docs:
            metadata = doc.metadata
            timestamp = metadata.get("timestamp", "Unknown time")
            formated_memories.append(f"[{timestamp}]\n{doc.page_content}")

        return "\n\n".join(formated_memories)

    def get_conversation_history(self) -> Dict[str, Any]:
        """Get the recent conversation history."""
        return self.short_term_memory.load_memory_variables({})
    
    def clear_short_term_memory(self) -> None:
        """Clear the short-term memory."""
        self.short_term_memory.clear()
    
    def get_all_memories(self) -> List[Document]:
        """Get all memories from the vector store."""
        # This is a simplified version - in practice, you'd implement pagination
        return self.vector_store.similarity_search("", k=1000)
