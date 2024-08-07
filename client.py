import os
import asyncio
import websockets
import json
import logging
import gradio as gr
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv("config.env")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class Client:
    def __init__(self):
        self.server_ip = ""
        self.name = ""
        self.backstory = ""
        self.websocket = None
        self.message_history: List[Tuple[str, str]] = []
        self.connected = False
        self.chat_history_component = None

    async def connect(self) -> Tuple[str, List[Tuple[str, str]]]:
        try:
            self.server_url = f"ws://{self.server_ip}:{WS_PORT}"
            logger.debug(f"Attempting to connect to {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            logger.debug("WebSocket connection established")
            
            setup_message = json.dumps({
                "type": "CHARACTER_SETUP",
                "name": self.name,
                "backstory": self.backstory
            })
            logger.debug(f"Sending setup message: {setup_message}")
            await self.websocket.send(setup_message)
            
            response = await self.websocket.recv()
            logger.debug(f"Received response: {response}")
            data = json.loads(response)
            
            if data["type"] == "SETUP_RESPONSE":
                self.message_history = data.get("history", [])
                self.connected = True
                asyncio.create_task(self.message_listener())
                return data["content"], self.message_history
            else:
                logger.error(f"Unexpected response: {data}")
                return f"Unexpected response: {data.get('content', 'Unknown error')}", []
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}", exc_info=True)
            return f"Failed to connect: {str(e)}", []
        
    async def handle_message(self, message):
        data = json.loads(message)
        if data['type'] == 'UPDATE_HISTORY':
            self.message_history.extend(data['messages'])
            await self.update_chat_history(self.message_history)
        elif data['type'] == 'SETUP_RESPONSE':
            self.message_history = data.get('history', [])
            await self.update_chat_history(self.message_history)

    async def send_action(self, action: str) -> None:
        if not self.connected:
            raise Exception("Not connected to the server.")
        
        try:
            message = json.dumps({
                "type": "CLIENT_MESSAGE",
                "name": self.name,
                "content": action
            })
            logger.debug(f"Sending action: {message}")
            await self.websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection lost")
            self.connected = False
            raise Exception("Connection lost. Please reconnect.")
        except Exception as e:
            logger.error(f"Error sending action: {str(e)}", exc_info=True)
            raise Exception(f"Error: {str(e)}")

    async def message_listener(self):
        while True:
            try:
                response = await self.websocket.recv()
                logger.debug(f"Received message: {response}")
                data = json.loads(response)
                if data['type'] == 'UPDATE_HISTORY':
                    self.message_history.extend(data['messages'])
                    await self.update_chat_history(self.message_history)
                elif data['type'] == 'SETUP_RESPONSE':
                    self.message_history = data.get('history', [])
                    await self.update_chat_history(self.message_history)
                else:
                    logger.warning(f"Received unexpected message type: {data['type']}")
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket connection closed")
                self.connected = False
                break
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON: {response}")
            except Exception as e:
                logger.error(f"Error in message listener: {str(e)}", exc_info=True)

    async def update_chat_history(self, history):
        if self.chat_history_component:
            await self.chat_history_component.update(value=history)
        else:
            logger.warning("Chat history component not set")

client = Client()

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# TaleWeaver - Client")

    with gr.Tab("Setup"):
        server_input = gr.Textbox(label="Enter server IP address (localhost by default)")
        name_input = gr.Textbox(label="Enter your character's name")
        backstory_input = gr.Textbox(label="Enter your character's backstory (optional)")
        connect_button = gr.Button("Connect to Server")
        connect_output = gr.Textbox(label="Connection Status")

    with gr.Tab("Play"):
        chat_history = gr.Chatbot(label="Chat History")
        client.chat_history_component = chat_history
        action_input = gr.Textbox(label="Enter your action")
        submit_button = gr.Button("Submit Action")

    async def connect_to_server(server, name, backstory):
        client.server_ip = server or "localhost"
        client.name = name
        client.backstory = backstory
        status, history = await client.connect()
        return status, history

    async def submit_action(action):
        if not client.connected:
            return "Not connected to the server. Please connect first."
        try:
            await client.send_action(action)
            return ""
        except Exception as e:
            return str(e)

    connect_button.click(connect_to_server, 
                         inputs=[server_input, name_input, backstory_input], 
                         outputs=[connect_output, chat_history])

    submit_button.click(submit_action, 
                        inputs=[action_input], 
                        outputs=[action_input])

if __name__ == "__main__":
    interface.launch()