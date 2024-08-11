import os
import json
import asyncio
import aiofiles
import logging
from typing import Optional, Dict, Set, List, Tuple
from quart import Quart, request, jsonify
from dotenv import load_dotenv
import gradio as gr
from ai_utils import AIClient
from game_logic import Player
from hypercorn.asyncio import serve
from hypercorn.config import Config
import threading
import websockets

# Get settings from the .env file
load_dotenv("config.env")
GROQ_API = os.getenv("GROQ_API")
MODEL = os.getenv("MODEL")
WS_PORT = int(os.getenv("WS_PORT", "8765"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = Quart(__name__)

class Server:
    def __init__(self, model: str):
        self.model = model
        self.players: Dict[str, Player] = {}
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.setting = ""
        self.current_session: str = ""
        self.message_history: List[Tuple[str, str]] = []
        self.ai_client = AIClient(api_key=GROQ_API, model=self.model)
        self.metadata_lock = asyncio.Lock()

    def generate_system_prompt(self) -> str:
        player_prompts = "\n".join([f"Player {player.name}'s character: {player.backstory}" 
                                    for player in self.players.values()])
        return f"You are guiding an adventure with a {self.setting} setting.\n{player_prompts}"

    async def add_player(self, name: str, backstory: str, ws: Optional[websockets.WebSocketServerProtocol]) -> str:
        if name not in self.players:
            self.players[name] = Player(name, backstory)
            if ws is not None:
                self.connections.add(ws)
            await self._save_metadata()
            logger.info(f"Player {name} added to the game.")
            return f"Player {name} added to the game."
        return f"Player {name} is already in the game."

    async def remove_player(self, ws: Optional[websockets.WebSocketServerProtocol]) -> str:
        if ws is not None and ws in self.connections:
            self.connections.remove(ws)
        for name, player in list(self.players.items()):
            if getattr(player, 'connection', None) == ws:
                del self.players[name]
                await self._save_metadata()
                return f"Player {name} removed from the game."
        return "Player not found."

    async def start_game(self, session_name: str, setting: str) -> str:
        self.current_session = session_name
        self.setting = setting
        self.players.clear()
        self.message_history.clear()
        new_system_prompt = self.generate_system_prompt()
        await self.ai_client.update_system_prompt(new_system_prompt)
        await self.ai_client.start_new_session(session_name)
        await self._save_metadata()
        return f"Game '{session_name}' started. You can now start your adventure!"

    async def broadcast_messages(self, messages: List[Tuple[str, str]]):
        if self.connections:
            message = json.dumps({
                "type": "UPDATE_HISTORY",
                "messages": messages
            })
            logger.debug(f"Broadcasting message to {len(self.connections)} connections: {message}")
            websockets.broadcast(self.connections, message)
        else:
            logger.warning("No connections to broadcast to")

    async def action(self, action: str, player_name: str) -> List[Tuple[str, str]]:
        logger.debug(f"Received action from {player_name}: {action}")
        try:
            response = await self.ai_client.generate(f"{player_name}: {action}")
            logger.debug(f"AI response: {response}")
            new_messages = [
                (player_name + ": " + action,
                 "AI: " + response)
            ]
            self.message_history.extend(new_messages)
            await self._save_metadata()
            await self.broadcast_messages(new_messages)
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
            "message_history": self.message_history
        }
        
        metadata_path = self._get_metadata_path(self.current_session)
        async with self.metadata_lock:
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))

    def _get_metadata_path(self, session_name: str) -> str:
        return os.path.join(self.ai_client.history_dir, f"{session_name}_metadata.json")

    async def load_session(self, session_name: str) -> str:
        self.current_session = session_name
        metadata_path = self._get_metadata_path(session_name)
        
        try:
            async with self.metadata_lock:
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata = json.loads(await f.read())
            
            self.players = {name: Player.from_dict(player_data) for name, player_data in metadata["players"].items()}
            self.setting = metadata["setting"]
            self.message_history = metadata.get("message_history", [])
            
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

async def websocket_handler(websocket, path):
    try:
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        while True:
            data = await websocket.recv()
            logger.debug(f"Received data: {data}")
            data = json.loads(data)
            if data["type"] == "CHARACTER_SETUP":
                add_response = await server.add_player(data["name"], data.get("backstory", ""), websocket)
                await websocket.send(json.dumps({
                    "type": "SETUP_RESPONSE",
                    "content": add_response,
                    "history": server.message_history
                }))
            elif data["type"] == "CLIENT_MESSAGE":
                player_name = data.get("name", "Unknown Player")
                action = data["content"]
                new_messages = await server.action(action, player_name)
                await server.broadcast_messages(new_messages)
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed for {websocket.remote_address}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON: {data}")
    except Exception as e:
        logger.error(f"Error in websocket handler: {str(e)}", exc_info=True)
    finally:
        await server.remove_player(websocket)

@app.route('/start_game', methods=['POST'])
async def start_game():
    data = await request.get_json()
    result = await server.start_game(data['session'], data['setting'])
    player_result = await server.add_player(data['name'], data['backstory'], None)
    return jsonify({"status": f"{result} {player_result}"})

@app.route('/load_session', methods=['POST'])
async def load_session():
    data = await request.get_json()
    result = await server.load_session(data['session_name'])
    return jsonify({"status": result, "history": server.message_history})

@app.route('/available_sessions', methods=['GET'])
async def available_sessions():
    return jsonify(server.get_available_sessions())

@app.route('/gm_action', methods=['POST'])
async def gm_action():
    data = await request.get_json()
    player_name = data.get('player_name', 'Game Master')
    new_messages = await server.action(data['action'], player_name)
    return jsonify({"messages": new_messages})

def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# TaleWeaver")
        
        with gr.Tab("Setup"):
            with gr.Column():
                session_name = gr.Textbox(label="Session Name")
            with gr.Row():
                setting_input = gr.Textbox(label="Enter the setting for the adventure")
                name_input = gr.Textbox(label="Enter your character's name")
                backstory_input = gr.Textbox(label="Enter your character's backstory")
            
            start_button = gr.Button("Start New Game")
            load_dropdown = gr.Dropdown(label="Load Previous Session", choices=server.get_available_sessions())
            load_button = gr.Button("Load Selected Session")
            setup_status = gr.Textbox(label="Setup Status")

        with gr.Tab("Game"):
            with gr.Column():
                chat_history = gr.Chatbot(label="Game History")
                action_input = gr.Textbox(label="Enter your action")
                game_status = gr.Textbox(label="Game Status", value="Not connected to a game session.")
                
        async def start_game_gradio(session, setting, name, backstory):
            result = await server.start_game(session, setting)
            player_result = await server.add_player(name, backstory, None)
            return f"{result}\n{player_result}", server.message_history, f"Connected to game: {session}"

        start_button.click(
            start_game_gradio,
            inputs=[session_name, setting_input, name_input, backstory_input],
            outputs=[setup_status, chat_history, game_status]
        )

        async def load_session_gradio(session_name):
            result = await server.load_session(session_name)
            return result, server.message_history, f"Loaded session: {session_name}"

        load_button.click(
            load_session_gradio,
            inputs=[load_dropdown],
            outputs=[setup_status, chat_history, game_status]
        )

        async def update_chat(history):
            return history

        chat_history.change(update_chat, inputs=[chat_history], outputs=[chat_history])

        async def chat_action(action, history):
            if not server.players:
                return history + [("System", "Please set up your character first.")], "", "Not in a game session."
            player_name = list(server.players.keys())[0]
            new_messages = await server.action(action, player_name)
            updated_history = history + new_messages
            await server.broadcast_messages(new_messages)
            return updated_history, "", "Action submitted and response received."

        action_input.submit(
            chat_action,
            inputs=[action_input, chat_history],
            outputs=[chat_history, action_input, game_status]
        )

    return interface

async def main():
    config = Config()
    config.bind = [f"0.0.0.0:{HTTP_PORT}"]
    
    gradio_app = create_gradio_interface()
    
    websocket_server = await websockets.serve(websocket_handler, "0.0.0.0", WS_PORT)
    
    def run_gradio():
        gradio_app.launch(server_port=7860, prevent_thread_lock=True)

    gradio_thread = threading.Thread(target=run_gradio)
    gradio_thread.start()
    
    await asyncio.gather(
        serve(app, config),
        websocket_server.wait_closed()
    )

if __name__ == "__main__":
    asyncio.run(main())