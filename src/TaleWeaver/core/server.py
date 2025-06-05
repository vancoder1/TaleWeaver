import os
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger

from utils.json_handler import JsonHandler
from core.model import Player
from deep_translator import GoogleTranslator
from core.ai_client_manager import AIClientManager
from core.session_manager import SessionManager

class Server:
    def __init__(self):
        self.json_handler = JsonHandler()
        self.character: Player = self._load_character_from_config()
        
        chat_history_base_dir = self.json_handler.get_setting('application_paths.chat_history_base_dir', 'data/chats')
        self.session_manager = SessionManager(self.json_handler, chat_history_base_dir)

        active_provider_name = self.json_handler.get_setting('ai_settings.active_provider')
        if not active_provider_name:
            raise ValueError("'ai_settings.active_provider' must be configured.")
        provider_config = self.json_handler.get_setting(f'ai_settings.providers.{active_provider_name}')
        if not provider_config:
            raise ValueError(f"Configuration for provider '{active_provider_name}' is missing.")

        initial_system_prompt = self.generate_system_prompt()
        try:
            self.ai_client = AIClientManager(
                provider_name=active_provider_name,
                provider_config=provider_config,
                system_prompt=initial_system_prompt,
                history_dir=chat_history_base_dir,
                json_handler=self.json_handler,
                session_id=self.session_manager.get_state()["name"] or "uninitialized_session"
            )
        except Exception as e:
            logger.critical(f"Failed to initialize AIClientManager: {e}", exc_info=True)
            raise
        logger.info(f"Server initialized with AIClientManager for provider: {active_provider_name}")

    def _load_character_from_config(self) -> Player:
        char_data = self.json_handler.get_setting("character", default={})
        player = Player.from_dict(char_data)
        if not char_data:
            self.json_handler.set_setting("character", player.to_dict())
        return player

    async def update_character_config(self, name: str, backstory: str) -> Tuple[bool, str]:
        if not name.strip(): return False, "Character name cannot be empty."
        self.character.name, self.character.backstory = name, backstory
        if self.json_handler.set_setting("character", self.character.to_dict()):
            await self.ai_client.update_system_prompt(self.generate_system_prompt())
            return True, f"Character '{name}' updated."
        return False, "Error saving character settings."

    def generate_system_prompt(self) -> str:
        base_prompt = self.json_handler.get_setting('ai_settings.system_prompt', "You are a helpful AI.")
        char_prompt = f"\nThe main character is {self.character.name}. Backstory: {self.character.backstory}"
        session_setting = self.session_manager.get_state()["setting"]
        setting_prompt = f"\nCurrent Game Setting: {session_setting}" if session_setting else "\nNo specific game setting."
        return f"{base_prompt}{char_prompt}{setting_prompt}"

    async def _translate_text(self, text: Optional[str], target_lang: str, source_lang: str) -> Optional[str]:
        if not text or target_lang == source_lang: return text
        try:
            sl = source_lang.lower() if source_lang == "zh-CN" else source_lang
            tl = target_lang.lower() if target_lang == "zh-CN" else target_lang
            return GoogleTranslator(source=sl, target=tl).translate(text) or text
        except Exception as e:
            logger.error(f"Translation error ({source_lang} to {target_lang}): {e}")
            return text

    async def start_game(self, session_name: str, setting: str, language_code: str) -> str:
        self.session_manager.start_session(session_name, setting, language_code)
        await self.ai_client.update_system_prompt(self.generate_system_prompt())
        await self.ai_client.start_new_session(session_name)
        await self.session_manager._save_metadata() # Initial save
        return f"Game '{session_name}' started. Setting: {setting}. Language: {language_code}."

    async def action(self, input_text: str, operation: str) -> List[Tuple[Optional[str], str]]:
        if not self.ai_client: return [(None, "AI System Error: Not initialized.")]
        if not self.session_manager.is_active(): return [(None, "System Error: No active game session.")]

        current_session_state = self.session_manager.get_state()
        current_lang_code = current_session_state["language"]
        char_name = self.character.name
        user_part_for_history: Optional[str] = None
        formatted_action_for_ai: Optional[str] = None

        try:
            if operation == "PlayerInput":
                if not input_text.strip(): return [(None, "Please enter an action.")]
                action_en = await self._translate_text(input_text, target_lang='en', source_lang=current_lang_code)
                formatted_action_for_ai = action_en or input_text
                user_part_for_history = f"{char_name}: {input_text}"
            elif operation == "Story":
                formatted_action_for_ai = f"Continue the story. (Context: {char_name} is the protagonist)."
            else: # Should not happen
                return [(None, f"System Error: Unknown operation: {operation}")]

            ai_response_en = "No action formulated for AI."
            if formatted_action_for_ai:
                ai_response_en = await self.ai_client.generate(formatted_action_for_ai)
            ai_response_en = ai_response_en or "AI did not provide a response."

            translated_ai_response = await self._translate_text(ai_response_en, target_lang=current_lang_code, source_lang='en')
            final_ai_response = translated_ai_response or ai_response_en

            await self.session_manager.add_history_and_save(user_part_for_history, final_ai_response)
            return [(user_part_for_history, final_ai_response)]
            
        except Exception as e:
            logger.error(f"Error processing {operation} for {char_name}: {e}", exc_info=True)
            error_msg = self.json_handler.get_setting("user_messages.error_action_processing", "AI: Error processing request.")
            return [(user_part_for_history if operation == "PlayerInput" else None, error_msg)]

    async def load_session(self, session_name: str, desired_language_key: str) -> Tuple[str, List[Tuple[Optional[str], str]]]:
        if not self.ai_client: 
            return "Error: AI System not initialized.", []

        success, status_msg, history = await self.session_manager.load_session(session_name, desired_language_key, self._translate_text)
        
        if success or "Metadata for" in status_msg : # if metadata loaded OR new metadata created after file not found
            await self.ai_client.load_session(session_name) # AI client loads its own history
            await self.ai_client.update_system_prompt(self.generate_system_prompt())
            final_msg = f"Loaded session: {session_name}. Language: {desired_language_key}."
            if "Metadata for" in status_msg and "not found" in status_msg: # File not found case
                 final_msg = (f"AI chat history for '{session_name}' loaded. "
                              f"New game metadata created. Language: {desired_language_key}.")
            return final_msg, history
        else: # Other load errors (corrupted, unexpected)
            # Attempt to load AI history even if metadata fails, if session_name is valid
            try:
                await self.ai_client.load_session(session_name)
                # Session manager state might be invalid, so we don't save metadata here
                # UI will show history from AI, but game state is ambiguous
                return f"AI history for '{session_name}' loaded, but game metadata failed: {status_msg}", [] # Empty UI history
            except Exception as ai_load_e:
                logger.error(f"AI session load also failed for '{session_name}' after metadata error: {ai_load_e}", exc_info=True)
                return f"No saved data found for session: '{session_name}'. Game metadata: {status_msg}", []


    def get_available_sessions(self) -> List[str]:
        return self.session_manager.get_available_sessions()
    
    def get_current_message_history(self) -> List[Tuple[Optional[str], str]]:
        return self.session_manager.get_state()["history"]

    async def update_config(self, new_config_values: Dict[str, Any]) -> str:
        changed_keys = []
        requires_ai_reinit = False
        current_active_provider = self.json_handler.get_setting('ai_settings.active_provider')

        for key, value in new_config_values.items():
            if self.json_handler.get_setting(key, None) != value:
                if self.json_handler.set_setting(key, value):
                    changed_keys.append(key)
                    newly_selected_provider = new_config_values.get('ai_settings.active_provider', current_active_provider)
                    provider_prefix = f'ai_settings.providers.{newly_selected_provider}.'
                    if key in ['ai_settings.active_provider', 'ai_settings.system_prompt'] or \
                       (key.startswith('ai_settings.providers.') and key.startswith(provider_prefix)):
                        requires_ai_reinit = True
                else: return f"Error updating config for {key}."
        if not changed_keys: return "Config up to date."

        if requires_ai_reinit:
            logger.info("Core AI settings changed. Re-initializing AIClientManager.")
            new_active_provider = self.json_handler.get_setting('ai_settings.active_provider')
            new_provider_config = self.json_handler.get_setting(f'ai_settings.providers.{new_active_provider}')
            if not new_active_provider or not new_provider_config:
                return f"Error: Cannot re-init AI. Missing config for '{new_active_provider}'."
            
            history_dir = self.json_handler.get_setting('application_paths.chat_history_base_dir', 'data/chats')
            session_id_for_reinit = self.session_manager.get_state()["name"] or "uninitialized_session"
            try:
                if self.ai_client: await self.ai_client.close()
                self.ai_client = AIClientManager(
                    new_active_provider, new_provider_config, self.generate_system_prompt(),
                    history_dir, self.json_handler, session_id_for_reinit
                )
                if self.session_manager.is_active():
                    await self.ai_client.load_session(session_id_for_reinit)
                return f"Config updated & AI re-initialized for: {', '.join(changed_keys)}."
            except Exception as e:
                 return f"Error re-initializing AI: {e}. Check logs."
        return f"Config updated for: {', '.join(changed_keys)}. AI settings applied."
    
    async def close_server_resources(self):
        logger.info("Closing server resources...")
        if self.ai_client:
            await self.ai_client.close()
        logger.info("Server resources closed.")