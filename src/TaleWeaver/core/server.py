import os
import json
import asyncio
import aiofiles
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger

from utils.ai_utils import AIClient
from utils.json_handler import JsonHandler
from utils import constants
from core.model import Player
from deep_translator import GoogleTranslator

class Server:
    def __init__(self):
        self.json_handler = JsonHandler()
        self.character: Player = self._load_character_from_config()
        self.setting: str = ""
        self.current_session: str = ""
        self.message_history: List[Tuple[str, str]] = []
        
        initial_system_prompt = self.generate_system_prompt()
        
        self.ai_client = AIClient(
            api_key=self.json_handler.get_setting('groq_api_key'),
            model=self.json_handler.get_setting('ai_settings.model'),
            system_prompt=initial_system_prompt
        )
        self.metadata_lock = asyncio.Lock()
        self.language = self.json_handler.get_setting('game_options.supported_languages.english', "en") # Default

    def _load_character_from_config(self) -> Player:
        char_data = self.json_handler.get_setting("character", default={})
        # Player.from_dict handles defaults if keys are missing
        player = Player.from_dict(char_data)
        if not char_data: # If no character data was found, save the new default
            logger.info("No character data in config. Creating and saving default character.")
            self.json_handler.set_setting("character", player.to_dict())
        else:
            logger.info(f"Loading character from config: {player.name}")
        return player

    async def update_character_config(self, name: str, backstory: str) -> Tuple[bool, str]:
        if not name.strip():
            return False, "Character name cannot be empty."
        
        self.character.name = name
        self.character.backstory = backstory
        
        if self.json_handler.set_setting("character", self.character.to_dict()):
            logger.info(f"Character settings updated in config: {self.character.name}")
            new_system_prompt = self.generate_system_prompt()
            await self.ai_client.update_system_prompt(new_system_prompt)
            return True, f"Character '{name}' updated successfully."
        else:
            logger.error("Failed to save character settings to config.")
            return False, "Error saving character settings."

    def generate_system_prompt(self) -> str:
        base_system_prompt = self.json_handler.get_setting('ai_settings.system_prompt', "You are a helpful AI.")
        
        # self.character is guaranteed to be initialized by __init__
        character_prompt_part = (
            f"\nThe main character is named {self.character.name}. "
            f"Character's backstory: {self.character.backstory}"
        )
        setting_prompt_part = f"\nCurrent Game Setting: {self.setting}" if self.setting else "\nNo specific game setting defined yet."
        
        full_prompt = f"{base_system_prompt}{character_prompt_part}{setting_prompt_part}"
        logger.debug(f"Generated system prompt snippet: {full_prompt[:150]}...")
        return full_prompt

    async def _translate_text(self, text: Optional[str], target_lang: str, source_lang: str) -> Optional[str]:
        if not text or target_lang == source_lang:
            return text
        try:
            # Consider creating translator instance once if performance becomes an issue for many small translations
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            return translated if translated is not None else text # Fallback to original if API returns None
        except Exception as e:
            logger.error(f"Translation error from {source_lang} to {target_lang} for text '{str(text)[:30]}...': {e}")
            return text # Fallback to original text on error

    async def start_game(self, session_name: str, setting: str, language: str) -> str:
        self.current_session = session_name
        self.setting = setting
        self.message_history.clear()
        self.language = language

        new_system_prompt = self.generate_system_prompt()
        await self.ai_client.update_system_prompt(new_system_prompt)
        await self.ai_client.start_new_session(session_name)
        await self._save_metadata()
        
        return f"Game '{session_name}' started for {self.character.name}. Setting: {setting}. Language: {language}. Begin your adventure!"

    async def action(self, action: str, mode: str) -> List[Tuple[str, str]]:
        original_action = action
        current_lang_code = self.language
        char_name = self.character.name # Assumed to be always valid

        try:
            action_en = await self._translate_text(action, target_lang='en', source_lang=current_lang_code)

            if mode == "Say":
                formatted_action = f"{char_name} says: \"{action_en}\""
            elif mode == "Do":
                formatted_action = f"{char_name} does: {action_en}"
            elif mode == "Story":
                 formatted_action = f"Continue the story. (Context: {char_name} is the protagonist)."
            else: # Should ideally be caught by UI dropdown, but as a safeguard:
                logger.warning(f"Unknown action mode: {mode}")
                return [(f"{char_name} ({mode}): {original_action}", "Error: Unknown action mode.")]

            ai_response_en = await self.ai_client.generate(formatted_action)
            translated_ai_response = await self._translate_text(ai_response_en, target_lang=current_lang_code, source_lang='en')
            
            if mode != "Story":
                new_messages = [(f"{char_name} ({mode}): {original_action}", f"AI: {translated_ai_response}")]
            else:
                new_messages = [("AI", translated_ai_response if translated_ai_response else "")]


            self.message_history.extend(new_messages)
            await self._save_metadata()
            return new_messages
        except Exception as e:
            logger.error(f"Error processing action for {char_name}, mode {mode}: {e}", exc_info=True)
            error_message = self.json_handler.get_setting("user_messages.error_action_processing", "AI: I'm sorry, but I encountered an error processing your action. Please try again.")
            return [(f"{char_name} ({mode}): {original_action}", error_message)]


    async def _save_metadata(self):
        if not self.current_session:
            return

        metadata = {
            "setting": self.setting,
            "message_history": self.message_history,
            "language": self.language
        }
        metadata_path = self._get_metadata_path(self.current_session)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        async with self.metadata_lock:
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=4, ensure_ascii=False))
        logger.debug(f"Session metadata saved for {self.current_session}")

    def _get_metadata_path(self, session_name: str) -> str:
        session_specific_data_dir = os.path.join(self.ai_client.history_dir, session_name)
        return os.path.join(session_specific_data_dir, f"{session_name}_game_metadata.json")

    async def load_session(self, session_name: str, desired_language_key: str) -> str:
        self.current_session = session_name
        metadata_path = self._get_metadata_path(session_name)
        desired_lang_code = constants.LANGUAGES.get(desired_language_key, "en")

        try:
            async with self.metadata_lock:
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
            
            self.setting = metadata.get("setting", "Default setting")
            session_saved_lang_code = metadata.get("language", "en")
            self.language = desired_lang_code 
            self.message_history = metadata.get("message_history", [])

            if self.message_history and session_saved_lang_code != self.language:
                logger.info(f"Translating message history from {session_saved_lang_code} to {self.language} for {session_name}")
                translated_history = []
                for user_msg_full, ai_msg_full in self.message_history:
                    ai_content_to_translate = ai_msg_full[len("AI: "):] if ai_msg_full.startswith("AI: ") else ai_msg_full
                    translated_ai_content = await self._translate_text(ai_content_to_translate, self.language, session_saved_lang_code)
                    new_ai_msg_full = f"AI: {translated_ai_content}" if ai_msg_full.startswith("AI: ") else translated_ai_content
                    translated_history.append((user_msg_full, new_ai_msg_full))
                self.message_history = translated_history

            await self.ai_client.load_session(session_name)
            new_system_prompt = self.generate_system_prompt()
            await self.ai_client.update_system_prompt(new_system_prompt)

            if desired_lang_code != session_saved_lang_code:
                 await self._save_metadata() # Save metadata if language choice changed for the session

            return f"Loaded session: {session_name} for {self.character.name}. Language: {desired_language_key}."

        except FileNotFoundError:
            logger.warning(f"No game metadata for {session_name}. Trying to load AI history.")
            try:
                await self.ai_client.load_session(session_name)
                self.setting = "Restored session (setting unknown)"
                self.message_history = [] 
                self.language = desired_lang_code
                await self.ai_client.update_system_prompt(self.generate_system_prompt())
                await self._save_metadata() # Create new metadata
                return f"AI chat history for '{session_name}' loaded for {self.character.name}. Metadata created. Language: {desired_language_key}."
            except Exception as ai_load_e:
                logger.error(f"AI session for {session_name} could not be loaded: {ai_load_e}")
                return f"No saved data found for: {session_name}. Start a new game."
        except json.JSONDecodeError:
            logger.error(f"Corrupted metadata for session: {session_name} at {metadata_path}")
            return f"Error loading session {session_name}: save file corrupted."
        except Exception as e:
            logger.error(f"Unexpected error loading session {session_name}: {e}", exc_info=True)
            return f"Unexpected error loading session {session_name}."

    def get_available_sessions(self) -> List[str]:
        history_dir = self.ai_client.history_dir
        if not os.path.isdir(history_dir): # Check if it's a directory
            return []
        try:
            return [d for d in os.listdir(history_dir) if os.path.isdir(os.path.join(history_dir, d))]
        except OSError as e: # Catch filesystem related errors
            logger.error(f"Error listing sessions in {history_dir}: {e}")
            return []

    async def update_config(self, new_config_values: Dict[str, str]) -> str:
        changed_keys = []
        core_ai_setting_changed = False

        for key, value in new_config_values.items():
            if self.json_handler.get_setting(key) != value:
                if self.json_handler.set_setting(key, value):
                    changed_keys.append(key)
                    if key in ['groq_api_key', 'ai_settings.model', 'ai_settings.system_prompt']:
                        core_ai_setting_changed = True
                else:
                    return f"Error updating configuration for {key}." # Early exit on failure

        if not changed_keys:
            return "Configuration is already up to date."

        if core_ai_setting_changed:
            logger.info(f"Core AI settings ({', '.join(changed_keys)}) updated. Re-initializing AIClient.")
            if hasattr(self.ai_client, 'close') and callable(self.ai_client.close):
                await self.ai_client.close()

            base_prompt_for_reinit = self.json_handler.get_setting('ai_settings.system_prompt')
            self.ai_client = AIClient(
                api_key=self.json_handler.get_setting('groq_api_key'),
                model=self.json_handler.get_setting('ai_settings.model'),
                system_prompt=base_prompt_for_reinit
            )
            current_full_system_prompt = self.generate_system_prompt()
            await self.ai_client.update_system_prompt(current_full_system_prompt)
            
            if self.current_session:
                 try:
                    await self.ai_client.load_session(self.current_session)
                    logger.info(f"Active session '{self.current_session}' history reloaded into new AIClient.")
                 except Exception as e:
                    logger.error(f"Failed to reload history for '{self.current_session}' into new AIClient: {e}")

        return f"Configuration updated for: {', '.join(changed_keys)}."
    
    async def close_server_resources(self):
        logger.info("Closing server resources...")
        if hasattr(self.ai_client, 'close') and callable(self.ai_client.close):
            await self.ai_client.close()
        logger.info("Server resources closed.")