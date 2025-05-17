from typing import Dict, Any, Optional
import logging

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
    """Chatbot with enhanced short-term and long-term memories."""

    def __init__(self, config: Optional[ChatbotConfig] = None):

        self.config = config or ChatbotConfig.from_env()

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

        Human: {user_input}
        AI Assistant:
        """

        return PromptTemplate(
            input_variables=["relevant_memories", "recent_history", "user_input"],
            template=template
        )

    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate a response to user input."""
        try:
            relevant_memories = self.memory.get_relevant_memories(user_input)

            recent_history = self.memory.get_conversation_history().get('history', '')


            prompt = self.prompt_template.format(
                relevant_memories=relevant_memories,
                recent_history=recent_history,
                user_input=user_input
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

    
