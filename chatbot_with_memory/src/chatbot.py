import os
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from langchain_google_genai import (
    ChatGoogleGenerativeAI, 
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from src.utils import ChatbotConfig
from src.memory import ChatbotMemory

logger = logging.getLogger(__name__)


class EnhancedChatbot:
    """Chatbot with enhanced memory capabilities."""

    def __init__(self, config: Optional[ChatbotConfig] = None):
        """Initialise the chatbot."""
        self.config = config or ChatbotConfig.from_env()
        self._setup_components()

    @staticmethod
    def format_conversation_history(history: Dict[str, Any]) -> str:
        """Format conversation history for display."""
        return history.get('history', '')

    def _setup_components(self) -> None:
        """Set up chatbot components."""

        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            google_api_key=self.config.google_api_key
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model
        )

        self.vector_store = Chroma(
            collection_name="conversation_memory",
            embedding_function=self.embeddings,
            persist_directory=self.config.chroma_persist_dir
        )

        self.memory = ChatbotMemory(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            memory_k=self.config.memory_k
        )

        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the chatbot."""
        template = """You are a helpful AI assistant with memory of past converstaions.
        
        Relevant past conversations:
        {relevant_memories}

        Recent conversation:
        {recent_history}

        Human: {input}
        AI Assistant:
        """

        return PromptTemplate(
            input_variables=["relevant_memories", "recent_history", "input"],
            template=template
        )

    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate a response to user input."""
        try:
            relevant_memories = self.memory.get_relevant_memories(user_input)

            recent_history = self.__class__.format_conversation_history(
                self.memory.get_conversation_history()
            )

            prompt = self.prompt_template.format(
                relevant_memories=relevant_memories,
                recent_history=recent_history,
                input=user_input
            )

            # Generate response
            response = self.llm.predict(prompt)

            # Add to memory
            self.memory.add_to_memory(user_input, response)

            return {
                "response": response,
                "relevant_memories": relevant_memories,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error generating response; {e}")
            return {
                "response": "I'm sorry, I encountered an error. Please try again.",
                "error": str(e),
                "success": False
            }
        
    def clear_memory(self) -> None:
        """Clear the chatbot's memory."""
        self.memory.clear_short_term_memory()
        logger.info("Short-term memory cleared")

    def get_conversation_history(self) -> str:
        """Get the formated conversation history."""
        return self.__class__.format_conversation_history(
            self.memory.get_conversation_history
        )
