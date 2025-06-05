import os
import json
import asyncio
import aiofiles
from typing import Dict, List, Tuple, Any, Optional, Callable, Awaitable
from loguru import logger

from utils.json_handler import JsonHandler
from utils import constants

class SessionManager:
    def __init__(self, json_handler: JsonHandler, chat_history_base_dir: str):
        self.json_handler = json_handler
        self.chat_history_base_dir = chat_history_base_dir
        os.makedirs(self.chat_history_base_dir, exist_ok=True)

        self.current_session_name: Optional[str] = None
        self.current_setting: str = ""
        self.current_message_history: List[Tuple[Optional[str], str]] = []
        self.current_language: str = self.json_handler.get_setting('game_options.supported_languages.english', "en")
        self.metadata_lock = asyncio.Lock()

    def is_active(self) -> bool:
        return bool(self.current_session_name)

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.current_session_name,
            "setting": self.current_setting,
            "language": self.current_language,
            "history": self.current_message_history,
        }

    def _get_metadata_path(self, session_name: str) -> str:
        session_specific_data_dir = os.path.join(self.chat_history_base_dir, session_name)
        return os.path.join(session_specific_data_dir, f"{session_name}_game_metadata.json")

    async def _save_metadata(self) -> None:
        if not self.current_session_name:
            logger.warning("Attempted to save metadata without an active session.")
            return

        metadata = {
            "setting": self.current_setting,
            "message_history": self.current_message_history,
            "language": self.current_language
        }
        metadata_path = self._get_metadata_path(self.current_session_name)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        async with self.metadata_lock:
            try:
                async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(metadata, indent=4, ensure_ascii=False))
                logger.debug(f"Session metadata saved for {self.current_session_name}")
            except Exception as e:
                logger.error(f"Failed to save metadata for {self.current_session_name}: {e}", exc_info=True)

    def start_session(self, session_name: str, setting: str, language: str) -> None:
        self.current_session_name = session_name
        self.current_setting = setting
        self.current_language = language
        self.current_message_history.clear()
        logger.info(f"Session manager started new session: '{session_name}'")

    async def load_session(self, session_name: str, desired_language_key: str,
                           translate_text_func: Callable[[Optional[str], str, str], Awaitable[Optional[str]]]) -> Tuple[bool, str, List[Tuple[Optional[str], str]]]:
        self.current_session_name = session_name
        metadata_path = self._get_metadata_path(session_name)
        desired_lang_code = constants.LANGUAGES.get(desired_language_key, "en")

        try:
            async with self.metadata_lock:
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
            
            self.current_setting = metadata.get("setting", "Default setting")
            session_saved_lang_code = metadata.get("language", "en")
            self.current_language = desired_lang_code
            
            loaded_history_raw = metadata.get("message_history", [])
            self.current_message_history = [tuple(item) for item in loaded_history_raw if isinstance(item, (list, tuple)) and len(item) == 2]
            
            if self.current_message_history and session_saved_lang_code != self.current_language:
                logger.info(f"Translating message history for '{session_name}' from '{session_saved_lang_code}' to '{self.current_language}'.")
                translated_history = []
                for user_msg, ai_msg in self.current_message_history:
                    translated_ai_msg = await translate_text_func(ai_msg, target_lang=self.current_language, source_lang=session_saved_lang_code)
                    translated_history.append((user_msg, translated_ai_msg or ai_msg or ""))
                self.current_message_history = translated_history

            if desired_lang_code != session_saved_lang_code:
                 await self._save_metadata()
            
            return True, f"Session '{session_name}' metadata loaded.", self.current_message_history

        except FileNotFoundError:
            logger.warning(f"No game metadata for session '{session_name}'. Session created with defaults.")
            self.current_setting = "Restored session (setting unknown)"
            self.current_message_history = []
            self.current_language = desired_lang_code
            await self._save_metadata() # Save new metadata for this "loaded" session
            return False, f"Metadata for '{session_name}' not found. New metadata created.", []
        except json.JSONDecodeError:
            logger.error(f"Corrupted metadata for session: '{session_name}'")
            self.current_session_name = None # Invalidate session
            return False, f"Error loading session '{session_name}': Metadata corrupted.", []
        except Exception as e:
            logger.error(f"Error loading session metadata '{session_name}': {e}", exc_info=True)
            self.current_session_name = None # Invalidate session
            return False, f"Unexpected error loading session metadata '{session_name}'.", []

    def get_available_sessions(self) -> List[str]:
        if not os.path.isdir(self.chat_history_base_dir):
            return []
        try:
            return [d for d in os.listdir(self.chat_history_base_dir) if os.path.isdir(os.path.join(self.chat_history_base_dir, d))]
        except OSError as e:
            logger.error(f"Error listing sessions in {self.chat_history_base_dir}: {e}")
            return []

    async def add_history_and_save(self, user_message: Optional[str], ai_message: str) -> None:
        if not self.current_session_name:
            logger.warning("Cannot add history, no active session.")
            return
        self.current_message_history.append((user_message, ai_message))
        await self._save_metadata()