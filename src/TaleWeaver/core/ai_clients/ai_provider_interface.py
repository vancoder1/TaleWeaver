from abc import ABC, abstractmethod
from typing import List, Optional, Any
from langchain_core.messages import BaseMessage
from loguru import logger
import os
import aiofiles # Moved import here

class AIProviderInterface(ABC):
    def __init__(self, model: str, system_prompt: str, session_id: str, history_dir: str,
                 api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        self.model_name = model
        self.current_system_prompt = system_prompt
        self.session_id = session_id
        self.history_dir = history_dir
        self.api_key = api_key
        self.base_url = base_url
        self.llm: Any = None
        self.chain: Any = None
        self.memory: Any = None
        self.additional_config = kwargs
        os.makedirs(self.history_dir, exist_ok=True)
        # _initialize_components called by subclasses

    @abstractmethod
    def _initialize_components(self, **kwargs: Any) -> None:
        """Initializes LLM, memory, chain."""
        pass

    @abstractmethod
    async def generate(self, input_text: str) -> str:
        """Generates AI response."""
        pass

    @abstractmethod
    async def update_system_prompt(self, new_full_prompt: str) -> None:
        """Updates system prompt."""
        pass

    @abstractmethod
    async def start_new_session(self, session_id: str) -> None:
        """Prepares for a new session."""
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> None:
        """Loads a previous session."""
        pass

    @abstractmethod
    async def get_conversation_history(self) -> List[BaseMessage]:
        """Retrieves conversation history."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes resources."""
        pass

    def _get_history_file_path(self) -> str:
        if not self.session_id: raise ValueError("Session ID must be set.")
        session_specific_dir = os.path.join(self.history_dir, self.session_id)
        return os.path.join(session_specific_dir, f"{self.session_id}_chat_history.json")

    async def _create_empty_history_file_async(self) -> None:
        file_path = self._get_history_file_path()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write('[]')
        except Exception as e:
            logger.error(f"Failed to create empty history file {file_path}: {e}", exc_info=True)
            raise