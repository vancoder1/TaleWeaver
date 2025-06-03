import os
import json
import asyncio
import aiofiles
from typing import Dict, List, Tuple, Any
from loguru import logger

from utils.ai_utils import AIClient
from utils.json_handler import JsonHandler
from utils import constants
from core.model import Player
from deep_translator import GoogleTranslator

class Server:
    def __init__(self):
        self.json_handler = JsonHandler()
        self.players: Dict[str, Player] = {}
        self.setting: str = ""
        self.current_session: str = ""
        self.message_history: List[Tuple[str, str]] = []
        self.ai_client = AIClient(
            api_key=self.json_handler.get_setting('groq_api_key'),
            model=self.json_handler.get_setting('ai_settings.model'),
            system_prompt=self.json_handler.get_setting('ai_settings.system_prompt')
        )
        self.metadata_lock = asyncio.Lock()
        self.language = "en"

    def generate_system_prompt(self) -> str:
        player_prompts = "\n".join([f"Player {player.name}'s character: {player.backstory}"
                                    for player in self.players.values()])
        base_system_prompt = self.json_handler.get_setting('ai_settings.system_prompt')
        return f"{base_system_prompt}\nSetting: {self.setting}\n{player_prompts}"

    async def add_player(self, name: str, backstory: str) -> str:
        if name not in self.players:
            self.players[name] = Player(name, backstory)
            await self._save_metadata() # metadata saving remains specific to session
            logger.info(f"Player {name} added to the game.")
            if self.current_session:
                new_system_prompt = self.generate_system_prompt()
                await self.ai_client.update_system_prompt(new_system_prompt)
            return f"Player {name} added to the game."
        return f"Player {name} is already in the game."

    async def remove_player(self, name: str) -> str:
        if name in self.players:
            del self.players[name]
            await self._save_metadata()
            logger.info(f"Player {name} removed from the game.")
            if self.current_session:
                new_system_prompt = self.generate_system_prompt()
                await self.ai_client.update_system_prompt(new_system_prompt)
            return f"Player {name} removed from the game."
        return "Player not found."

    async def start_game(self, session_name: str, setting: str, language: str) -> str:
        self.current_session = session_name
        self.setting = setting
        self.message_history.clear()
        self.language = language

        new_system_prompt = self.generate_system_prompt()
        await self.ai_client.update_system_prompt(new_system_prompt)
        await self.ai_client.start_new_session(session_name)
        await self._save_metadata()
        return f"Game '{session_name}' started. Setting: {setting}. You can now add players and begin your adventure!"

    async def action(self, action: str, player_name: str, mode: str) -> List[Tuple[str, str]]:
        try:
            original_action = action
            current_lang_code = self.language

            if current_lang_code != "en":
                try:
                    translator_to_en = GoogleTranslator(source=current_lang_code, target='en')
                    action_en = translator_to_en.translate(action)
                    if action_en is None:
                        logger.error(f"Translation to English failed for: {action}")
                        action_en = action
                except Exception as trans_e:
                    logger.error(f"Translator (to EN) error: {trans_e}")
                    action_en = action
            else:
                action_en = action

            if mode == "Say":
                formatted_action = f"{player_name} says: \"{action_en}\""
            elif mode == "Do":
                formatted_action = f"{player_name} does: {action_en}"
            elif mode == "Story":
                 formatted_action = f"Continue the story without player input."
            else:
                logger.warning(f"Unknown action mode: {mode}")
                return [(player_name, "Error: Unknown action mode.")]

            ai_response_en = await self.ai_client.generate(formatted_action)

            translated_ai_response = ai_response_en
            if current_lang_code != "en":
                try:
                    translator_from_en = GoogleTranslator(source='en', target=current_lang_code)
                    translated_ai_response = translator_from_en.translate(ai_response_en)
                    if translated_ai_response is None:
                        logger.error(f"Translation from English failed for AI response.")
                        translated_ai_response = ai_response_en
                except Exception as trans_e:
                    logger.error(f"Translator (from EN) error: {trans_e}")
                    translated_ai_response = ai_response_en

            if mode != "Story":
                new_messages = [(f"{player_name} ({mode}): {original_action}", f"AI: {translated_ai_response}")]
            else:
                new_messages = [("AI", translated_ai_response)]

            self.message_history.extend(new_messages)
            await self._save_metadata()
            return new_messages
        except Exception as e:
            logger.error(f"Error processing action for player {player_name}, mode {mode}: {e}", exc_info=True)
            return [(f"{player_name} ({mode}): {original_action}", "AI: I'm sorry, but I encountered an error processing your action. Please try again.")]

    async def _save_metadata(self):
        if not self.current_session:
            return

        metadata = {
            "players": {name: player.to_dict() for name, player in self.players.items()},
            "setting": self.setting,
            "message_history": self.message_history,
            "language": self.language
        }

        metadata_path = self._get_metadata_path(self.current_session)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        async with self.metadata_lock:
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=4, ensure_ascii=False))
        logger.debug(f"Session metadata saved for {self.current_session} to {metadata_path}")

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

            self.players = {name: Player.from_dict(player_data) for name, player_data in metadata.get("players", {}).items()}
            self.setting = metadata.get("setting", "Default setting")
            session_saved_lang_code = metadata.get("language", "en")
            self.language = desired_lang_code
            self.message_history = metadata.get("message_history", [])

            if self.message_history and session_saved_lang_code != self.language:
                logger.info(f"Translating message history from {session_saved_lang_code} to {self.language} for session {session_name}")
                translated_history = []
                for user_msg_full, ai_msg_full in self.message_history:
                    ai_content_to_translate = ai_msg_full
                    if ai_msg_full.startswith("AI: "):
                        ai_content_to_translate = ai_msg_full[len("AI: "):]
                    try:
                        translator = GoogleTranslator(source=session_saved_lang_code, target=self.language)
                        translated_ai_content = translator.translate(ai_content_to_translate)
                        if translated_ai_content is None: translated_ai_content = ai_content_to_translate
                    except Exception as trans_e:
                        logger.error(f"Error translating history message: {trans_e}")
                        translated_ai_content = ai_content_to_translate
                    new_ai_msg_full = f"AI: {translated_ai_content}" if ai_msg_full.startswith("AI: ") else translated_ai_content
                    translated_history.append((user_msg_full, new_ai_msg_full))
                self.message_history = translated_history

            await self.ai_client.load_session(session_name)
            new_system_prompt = self.generate_system_prompt()
            await self.ai_client.update_system_prompt(new_system_prompt)

            if desired_lang_code != session_saved_lang_code:
                await self._save_metadata()

            logger.info(f"Loaded game session: {session_name} with {len(self.players)} players. Language set to {self.language}.")
            return f"Loaded session: {session_name} with {len(self.players)} players."
        except FileNotFoundError:
            logger.warning(f"No saved game metadata found for: {session_name} at {metadata_path}.")
            try:
                await self.ai_client.load_session(session_name)
                self.players.clear()
                self.setting = "Newly restored from chat history"
                self.message_history = []
                self.language = desired_lang_code
                new_system_prompt = self.generate_system_prompt()
                await self.ai_client.update_system_prompt(new_system_prompt)
                await self._save_metadata()
                return f"AI chat history for '{session_name}' loaded. Game metadata was missing, starting fresh with this history. Please add players."
            except Exception as ai_load_e:
                logger.error(f"AI session could not be loaded for {session_name} either: {ai_load_e}")
                return f"No saved data found for: {session_name}. Please start a new game."
        except json.JSONDecodeError:
            logger.error(f"Error loading game metadata for session: {session_name}. File may be corrupted: {metadata_path}")
            return f"Error loading session: {session_name}. The save file may be corrupted."
        except Exception as e:
            logger.error(f"Unexpected error loading session {session_name}: {e}", exc_info=True)
            return f"An unexpected error occurred while loading session {session_name}."

    def get_available_sessions(self) -> List[str]:
        history_dir = self.ai_client.history_dir
        if not os.path.exists(history_dir):
            return []
        return [d for d in os.listdir(history_dir) if os.path.isdir(os.path.join(history_dir, d))]

    async def update_config(self, new_config_values: Dict[str, str]) -> str:
        changed_keys = []
        for key, value in new_config_values.items():
            if self.json_handler.get_setting(key) != value:
                if self.json_handler.set_setting(key, value):
                    changed_keys.append(key)
                else:
                    logger.error(f"Failed to set config key {key} using JsonHandler.")
                    return f"Error updating configuration for {key}."

        if changed_keys:
            logger.info(f"Configuration keys updated: {', '.join(changed_keys)}. Re-initializing AIClient.")
            self.ai_client = AIClient(
                api_key=self.json_handler.get_setting('groq_api_key'),
                model=self.json_handler.get_setting('ai_settings.model'),
                system_prompt=self.json_handler.get_setting('ai_settings.system_prompt')
            )
            if self.current_session:
                logger.info("Active game session found, regenerating and applying system prompt to new AIClient.")
                new_system_prompt = self.generate_system_prompt()
                await self.ai_client.update_system_prompt(new_system_prompt)
            return f"Configuration updated successfully for: {', '.join(changed_keys)}."
        return "Configuration is already up to date. No changes made."