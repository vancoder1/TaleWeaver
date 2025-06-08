from loguru import logger
from typing import Dict, Any, List, Type

from core.ai_clients.ai_provider_interface import AIProviderInterface
from core.ai_clients.groq_ai_client import GroqAIClient
# from core.ai_clients.gemini_ai_client import GeminiAIClient
# from core.ai_clients.ollama_ai_client import OllamaAIClient
# from core.ai_clients.openai_ai_client import OpenAIAIClient

from utils.json_handler import JsonHandler

CLIENT_REGISTRY: Dict[str, Type[AIProviderInterface]] = {
    "groq": GroqAIClient,
    # "gemini": GeminiAIClient, # Add when implemented
    # "ollama": OllamaAIClient, # Add when implemented
    # "openai": OpenAIAIClient, # Add when implemented
}

class AIClientManager:
    def __init__(self, provider_name: str, provider_config: Dict[str, Any], system_prompt: str,
                 history_dir: str, json_handler: JsonHandler, session_id: str = "default_manager_session"):
        self.provider_name = provider_name.lower()
        self.provider_config = provider_config
        self.current_system_prompt = system_prompt
        self.history_dir = history_dir
        self.json_handler = json_handler
        self.session_id = session_id

        self.active_client: AIProviderInterface = self._create_provider_client()
        logger.info(f"AIClientManager initialized with '{self.provider_name}' for session '{self.session_id}'.")

    def _create_provider_client(self) -> AIProviderInterface:
        model = self.provider_config.get("model")
        if not model: raise ValueError(f"Model not configured for provider '{self.provider_name}'.")
        
        provider_args = {
            "model": model,
            "system_prompt": self.current_system_prompt,
            "session_id": self.session_id,
            "history_dir": self.history_dir,
            "api_key": self.provider_config.get("api_key"),
            "base_url": self.provider_config.get("base_url"),
        }

        client_class = CLIENT_REGISTRY.get(self.provider_name)
        if client_class:
            return client_class(**provider_args)
        
        raise ValueError(f"Unsupported or unconfigured AI provider: {self.provider_name}")

    async def generate(self, input_text: str) -> str:
        return await self.active_client.generate(input_text)

    async def update_system_prompt(self, new_full_prompt: str) -> None:
        self.current_system_prompt = new_full_prompt
        await self.active_client.update_system_prompt(new_full_prompt)

    async def start_new_session(self, session_id: str) -> None:
        self.session_id = session_id
        await self.active_client.start_new_session(session_id)

    async def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        await self.active_client.load_session(session_id)

    async def get_conversation_history(self) -> List[Any]:
        return await self.active_client.get_conversation_history()

    async def close(self) -> None:
        if self.active_client:
            await self.active_client.close()
        logger.info(f"AIClientManager for '{self.provider_name}' closed.")

    def get_current_provider_name(self) -> str: return self.provider_name
    def get_current_session_id(self) -> str: return self.session_id
    def get_current_system_prompt(self) -> str: return self.current_system_prompt