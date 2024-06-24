import os
from groq import Groq
import socket
import json
import threading
import requests
from dotenv import load_dotenv, dotenv_values
from ai_utils import AIClient
from game_logic import Player

# Get settings from the .env file
load_dotenv("config.env")
GROQ_API = os.getenv("GROQ_API")
MODEL = os.getenv("MODEL")

class Server:
    def __init__(self, model):
        self.host_name = socket.gethostbyname("0.0.0.0")
        self.port = 5555
        self.model = model
        self.system_prompt = ""
        self.clients = {}
        self.server_player = Player()
        self.ai_client = AIClient(api_key=GROQ_API, model=model, system_prompt=self.system_prompt)
        self.lock = threading.Lock()  # Lock to manage input blocking

    def handle_client(self, client_socket):
        try:
            while True:
                message = client_socket.recv(8192).decode("utf-8")
                if message:
                    data = json.loads(message)
                    if data["type"] == "CHARACTER_SETUP":
                        player = Player(data["name"], data["backstory"])
                        self.clients[client_socket] = player
                        self.update_system_prompt()
                        print(f"Player {str(data['name'])} added")  # Debug
                    elif data["type"] == "CLIENT_MESSAGE":
                        with self.lock:
                            player = self.clients.get(client_socket)
                            if player:
                                print(f"\r\n{str(data['content'])}")
                            else:
                                print(f"\r\nUnknown Player: {str(data['content'])}")
                            response = self.ai_client.generate_text(str(data["content"]))
                            print(f"\nAI: {response}")
                            client_socket.send(json.dumps({"type": "AI_RESPONSE", "content": response}).encode("utf-8"))
        except Exception as e:
            print(f"Error: {e}")
            client_socket.close()
            if client_socket in self.clients:
                del self.clients[client_socket]

    def broadcast(self, message):
        for client_socket in self.clients.keys():
            client_socket.send(json.dumps(message).encode("utf-8"))

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host_name, self.port))
        server_socket.listen(5)
        print("Server started. Waiting for connections...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler.start()
            server_input_handler = threading.Thread(target=self.server_input)
            server_input_handler.start()


    def server_input(self):
        while True:
            prompt = f"\n{self.server_player.name}: "
            prompt += input(prompt)
            print("\n" + prompt)
            self.broadcast({"type": "SERVER_MESSAGE",
                            "content": prompt})
            response = self.ai_client.generate_text(prompt)
            print(f"\nAI: {response}")
            self.broadcast({"type": "AI_RESPONSE", 
                            "content": response})

    def update_system_prompt(self):
        client_prompts = ""
        for client_socket, player in self.clients.items():
            client_prompts += f"\nThe second player's character is {player.name}."
            if player.backstory:
                client_prompts += f" The backstory of the second character is: {player.backstory}."
        self.system_prompt += client_prompts
        self.ai_client.system_prompt = self.system_prompt

    def choose_mode(self):
        while True:
            mode = input("Choose mode (1 for solo, 2 for coop): ").strip()
            if mode in ["1", "2"]:
                break
            else:
                print("Invalid input. Please enter 1 or 2.")

        setting = input("Enter the setting for the adventure (e.g., fantasy, sci-fi): ").strip()
        self.server_player.name = input("Enter your character's name: ").strip()
        self.server_player.backstory = input("Enter your character's backstory (optional): ").strip()

        base_prompt = f"""
        You are guiding an immersive adventure with {setting} setting.
        """

        if mode == "1":
            self.system_prompt = base_prompt + f"""
            The main character is {self.server_player.name}.
            """
            if self.server_player.backstory:
                self.system_prompt += f"The backstory of the character is: {self.server_player.backstory}."

            self.system_prompt += base_prompt + f"""
            Respond creatively to the player's actions, providing vivid descriptions that shape the narrative.
            Keep the tone engaging and the story creative.
            """
            self.ai_client.system_prompt = self.system_prompt
            self.play_solo()

        elif mode == "2":    
            self.system_prompt = f"""
            Facilitate a cooperative adventure between multiple players. 
            Respond to their actions providing vivid descriptions that shape the shared narrative. 
            Keep the tone engaging and the story creative
            """
            self.system_prompt += base_prompt + f"""
            The first player's character is {self.server_player.name}.
            """
            if self.server_player.backstory:
                self.system_prompt += f" The backstory of the first character is: {self.server_player.backstory}."
            self.start_server()

    def play_solo(self):
        while True:
            prompt = input(f"\n{self.server_player.name}: ")
            response = self.ai_client.generate_text(prompt)
            print(f"\nAI: {response}")

if __name__ == "__main__":
    server = Server(MODEL)
    server.choose_mode()