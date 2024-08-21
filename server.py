import os
import json
import asyncio
import threading
import aiofiles
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import gradio as gr
from ai_utils import AIClient
from game_logic import Player
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv("config.env")
GROQ_API = os.getenv("GROQ_API")
MODEL = os.getenv("MODEL")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

LANGUAGES = {
    "english": "en",
    "russian": "ru",
    "chinese": "zh-CN",
    "japanese": "ja",
    "french": "fr",
    "german": "de"
}

class Server:
    def __init__(self, model: str):
        self.model = model
        self.players: Dict[str, Player] = {}
        self.setting = ""
        self.current_session: str = ""
        self.message_history: List[Tuple[str, str]] = []
        self.ai_client = AIClient(api_key=GROQ_API, model=self.model)
        self.metadata_lock = asyncio.Lock()
        self.translation_enabled = False
        self.language = "en"

    def generate_system_prompt(self) -> str:
        player_prompts = "\n".join([f"Player {player.name}'s character: {player.backstory}" 
                                    for player in self.players.values()])
        return f"You are guiding an adventure with a {self.setting} setting.\n{player_prompts}"

    async def add_player(self, name: str, backstory: str) -> str:
        if name not in self.players:
            self.players[name] = Player(name, backstory)
            await self._save_metadata()
            logger.info(f"Player {name} added to the game.")
            return f"Player {name} added to the game."
        return f"Player {name} is already in the game."

    async def remove_player(self, name: str) -> str:
        if name in self.players:
            del self.players[name]
            await self._save_metadata()
            return f"Player {name} removed from the game."
        return "Player not found."

    async def start_game(self, session_name: str, setting: str, language: str, translation_enabled: bool) -> str:
        self.current_session = session_name
        self.setting = setting
        self.players.clear()
        self.message_history.clear()
        self.language = language
        self.translation_enabled = translation_enabled
        new_system_prompt = self.generate_system_prompt()
        await self.ai_client.update_system_prompt(new_system_prompt)
        await self.ai_client.start_new_session(session_name)
        await self._save_metadata()
        return f"Game '{session_name}' started. You can now start your adventure!"

    async def action(self, action: str, player_name: str) -> List[Tuple[str, str]]:
        try:
            original_action = action
            if self.translation_enabled and self.language != "en":
                translator = GoogleTranslator(source=self.language, target='en')
                action = translator.translate(action)
            
            response = await self.ai_client.generate(f"{player_name}: {action}")
            
            if self.translation_enabled and self.language != "en":
                translator = GoogleTranslator(source='en', target=self.language)
                translated_response = translator.translate(response)
            else:
                translated_response = response
            
            new_messages = [(player_name + ": " + original_action, "AI: " + translated_response)]
            self.message_history.extend(new_messages)
            await self._save_metadata()
            return new_messages
        except Exception as e:
            logger.error(f"Error processing action: {str(e)}")
            return [(player_name + ": " + action, "AI: I'm sorry, but I encountered an error. Please try again.")]

    async def _save_metadata(self):
        if not self.current_session:
            return

        metadata = {
            "players": {name: player.to_dict() for name, player in self.players.items()},
            "setting": self.setting,
            "message_history": self.message_history,
            "language": self.language,
            "translation_enabled": self.translation_enabled
        }
        
        metadata_path = self._get_metadata_path(self.current_session)
        async with self.metadata_lock:
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))

    def _get_metadata_path(self, session_name: str) -> str:
        return os.path.join(self.ai_client.history_dir, f"{session_name}_metadata.json")

    async def load_session(self, session_name: str, language: str, translation_enabled: bool) -> str:
        self.current_session = session_name
        metadata_path = self._get_metadata_path(session_name)
        
        try:
            async with self.metadata_lock:
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata = json.loads(await f.read())
            
            self.players = {name: Player.from_dict(player_data) for name, player_data in metadata["players"].items()}
            self.setting = metadata["setting"]
            self.language = language
            self.translation_enabled = translation_enabled
            
            self.message_history = metadata.get("message_history", [])
            if self.language != metadata.get("language") or self.translation_enabled != metadata.get("translation_enabled", False):
                # Translate message history if language settings have changed
                translator = GoogleTranslator(source=metadata.get("language", "en"), target=self.language)
                self.message_history = [(msg[0], translator.translate(msg[1])) for msg in self.message_history]
            
            await self.ai_client.load_session(session_name)
            new_system_prompt = self.generate_system_prompt()
            await self.ai_client.update_system_prompt(new_system_prompt)
            
            return f"Loaded session: {session_name} with {len(self.players)} players."
        except FileNotFoundError:
            return f"No saved session found for: {session_name}. Please start a new game."
        except json.JSONDecodeError:
            return f"Error loading session: {session_name}. The save file may be corrupted."

    def get_available_sessions(self) -> List[str]:
        history_dir = self.ai_client.history_dir
        return [f.split('_metadata.json')[0] for f in os.listdir(history_dir) if f.endswith('_metadata.json')]

server = Server(MODEL)

async def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# TaleWeaver")
        
        with gr.Tab("Setup"):
            with gr.Column():
                session_name = gr.Textbox(label="Session Name")
            with gr.Row():
                setting_input = gr.Textbox(label="Enter the setting for the adventure")
                name_input = gr.Textbox(label="Enter your character's name")
                backstory_input = gr.Textbox(label="Enter your character's backstory")
            with gr.Column():
                language_dropdown = gr.Dropdown(label="Select Language", choices=list(LANGUAGES.keys()), value="english")
                translation_checkbox = gr.Checkbox(label="Enable Translation", value=False)
            
            start_button = gr.Button("Start New Game")
            load_dropdown = gr.Dropdown(label="Load Previous Session", choices=server.get_available_sessions())
            load_button = gr.Button("Load Selected Session")
            setup_status = gr.Textbox(label="Setup Status")

        with gr.Tab("Game"):
            with gr.Column():
                chat_history = gr.Chatbot(label="Game History")
                action_input = gr.Textbox(label="Enter your action")
                game_status = gr.Textbox(label="Game Status", value="Not connected to a game session.")
                
        async def start_game_gradio(session, setting, name, backstory, language, translation_enabled):
            result = await server.start_game(session, setting, LANGUAGES[language], translation_enabled)
            player_result = await server.add_player(name, backstory)
            return f"{result}\n{player_result}", server.message_history, f"Connected to game: {session}"

        start_button.click(
            start_game_gradio,
            inputs=[session_name, setting_input, name_input, backstory_input, language_dropdown, translation_checkbox],
            outputs=[setup_status, chat_history, game_status]
        )

        async def load_session_gradio(session_name, language, translation_enabled):
            result = await server.load_session(session_name, LANGUAGES[language], translation_enabled)
            return result, server.message_history, f"Loaded session: {session_name}"

        load_button.click(
            load_session_gradio,
            inputs=[load_dropdown, language_dropdown, translation_checkbox],
            outputs=[setup_status, chat_history, game_status]
        )

        async def chat_action(action, history):
            if not server.players:
                return history + [("System", "Please set up your character first.")], "", "Not in a game session."
            player_name = list(server.players.keys())[0]
            new_messages = await server.action(action, player_name)
            updated_history = history + new_messages
            return updated_history, "", "Action submitted and response received."

        action_input.submit(
            chat_action,
            inputs=[action_input, chat_history],
            outputs=[chat_history, action_input, game_status]
        )

    return interface

async def main(): 
    gradio_app = await create_gradio_interface()
    
    # Create an event to signal when Gradio is ready
    gradio_ready = asyncio.Event()

    def run_gradio():
        gradio_app.launch(prevent_thread_lock=True)
        gradio_ready.set()  # Signal that Gradio is ready

    # Start Gradio in a separate thread
    threading.Thread(target=run_gradio, daemon=True).start()

    # Wait for Gradio to be ready
    await gradio_ready.wait()

if __name__ == "__main__":
    asyncio.run(main())