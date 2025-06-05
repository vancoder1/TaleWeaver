from loguru import logger
from typing import Dict, Any, List, Type

from .ai_clients.ai_provider_interface import AIProviderInterface
from .ai_clients.groq_ai_client import GroqAIClient
# Import other clients as they are implemented
# from .ai_clients.gemini_ai_client import GeminiAIClient
# from .ai_clients.ollama_ai_client import OllamaAIClient
# from .ai_clients.openrouter_ai_client import OpenRouterAIClient

from utils.json_handler import JsonHandler

CLIENT_REGISTRY: Dict[str, Type[AIProviderInterface]] = {
    "groq": GroqAIClient,
    # "gemini": GeminiAIClient, # Add when implemented
    # "ollama": OllamaAIClient, # Add when implemented
    # "openrouter": OpenRouterAIClient, # Add when implemented
}

class PlaceholderClient(AIProviderInterface):
    """Minimal placeholder for unimplemented AI providers."""
    def __init__(self, provider_name_for_error: str, **kwargs):
        super().__init__(model="placeholder", system_prompt="", session_id="placeholder", history_dir="data/tmp")
        self.error_message = f"Error: AI Provider '{provider_name_for_error}' is configured but not implemented."
        logger.error(self.error_message)
    def _initialize_components(self, **kwargs: Any) -> None: pass
    async def generate(self, input_text: str) -> str: return self.error_message
    async def update_system_prompt(self, new_full_prompt: str) -> None: pass
    async def start_new_session(self, session_id: str) -> None: pass
    async def load_session(self, session_id: str) -> None: pass
    async def get_conversation_history(self) -> List[Any]: return []
    async def close(self) -> None: pass

class AIClientManager:
    def __init__(self, provider_name: str, provider_config: Dict[str, Any], system_prompt: str,
                 history_dir: str, json_handler: JsonHandler, session_id: str = "default_manager_session"):
        self.provider_name = provider_name.lower()
        self.current_system_prompt = system_prompt # Stored for re-init, actual prompt managed by client
        self.session_id = session_id # Stored for re-init, actual session_id managed by client

        # These are passed to the client constructor and used by it
        self.provider_config = provider_config
        self.history_dir = history_dir
        self.json_handler = json_handler # Used for shared AI settings like token limits

        self.active_client: AIProviderInterface = self._create_provider_client(
            self.provider_name, self.provider_config, system_prompt, 
            self.history_dir, self.json_handler, session_id
        )
        logger.info(f"AIClientManager initialized with '{self.provider_name}' for session '{session_id}'.")

    def _create_provider_client(self, provider_name: str, provider_config: Dict[str, Any], 
                                system_prompt: str, history_dir: str, 
                                json_handler: JsonHandler, session_id: str) -> AIProviderInterface:
        
        model = provider_config.get("model")
        if not model: raise ValueError(f"Model not configured for provider '{provider_name}'.")

        shared_ai_settings = json_handler.get_setting("ai_settings", {})
        
        provider_args = {
            "model": model,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "history_dir": history_dir,
            "api_key": provider_config.get("api_key"),
            "base_url": provider_config.get("base_url"),
            "conversation_summary_max_token_limit": shared_ai_settings.get("conversation_summary_max_token_limit", 450),
            **(provider_config.get("additional_params", {}))
        }

        client_class = CLIENT_REGISTRY.get(provider_name)
        if client_class:
            if provider_name == "ollama" and 'api_key' in provider_args and not provider_config.get("api_key"):
                del provider_args['api_key']
            return client_class(**provider_args)
        
        # Fallback for configured but not registered providers
        known_providers_in_config = json_handler.get_setting("ai_settings.providers", {}).keys()
        if provider_name in [p.lower() for p in known_providers_in_config]:
             return PlaceholderClient(provider_name_for_error=provider_name, **provider_args)
        
        raise ValueError(f"Unsupported or unconfigured AI provider: {provider_name}")

    async def generate(self, input_text: str) -> str:
        return await self.active_client.generate(input_text)

    async def update_system_prompt(self, new_full_prompt: str) -> None:
        self.current_system_prompt = new_full_prompt # Update manager's copy for potential re-init
        await self.active_client.update_system_prompt(new_full_prompt)

    async def start_new_session(self, session_id: str) -> None:
        self.session_id = session_id # Update manager's copy
        await self.active_client.start_new_session(session_id)

    async def load_session(self, session_id: str) -> None:
        self.session_id = session_id # Update manager's copy
        await self.active_client.load_session(session_id)

    async def get_conversation_history(self) -> List[Any]:
        return await self.active_client.get_conversation_history()

    async def close(self) -> None:
        if self.active_client:
            await self.active_client.close()
        logger.info(f"AIClientManager for '{self.provider_name}' closed.")

    # Methods to get current state if needed by Server for re-initialization
    def get_current_provider_name(self) -> str: return self.provider_name
    def get_current_session_id(self) -> str: return self.session_id
    def get_current_system_prompt(self) -> str: return self.current_system_prompt