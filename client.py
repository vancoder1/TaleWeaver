import socket
import time
import json
import threading
from colorama import Fore
from game_logic import Player

class Client:
    def __init__(self):
        self.server_ip = "127.0.0.1"
        self.port = 5555
        self.client_player = Player()
        self.lock = threading.Lock()  # Lock to manage input blocking

    def receive_messages(self, client_socket):
        try:
            while True:
                message = client_socket.recv(8192).decode("utf-8")
                if message:
                    with self.lock:
                        data = json.loads(message)
                        if data["type"] == "SERVER_MESSAGE":
                            print(Fore.BLUE + f"\r\n{str(data['content'])}")
                        if data["type"] == "AI_RESPONSE":
                            print(Fore.WHITE + f"\nAI: {str(data['content'])}")
        except Exception as e:
            print(f"Error: {e}")
            client_socket.close()

    def client_input(self, client_socket):
        while True:
            prompt = f"\n{self.client_player.name}: "
            prompt += input(Fore.RED + prompt)
            client_socket.send(json.dumps({"type": "CLIENT_MESSAGE", "content": prompt}).encode('utf-8'))

    def start_client(self):
        while True:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect((self.server_ip, self.port))
                print("Connected to the server.")
                self.set_character(client)  # Prompt for character setup immediately after connecting

                receive_thread = threading.Thread(target=self.receive_messages, args=(client,))
                input_thread = threading.Thread(target=self.client_input, args=(client,))
                receive_thread.start()
                input_thread.start()
                break  # Exit the loop once connected
            except socket.error as e:
                print(f"Connection failed: {e}. Retrying in 3 seconds...")
                time.sleep(3)  # Wait for 3 seconds before retrying

    def set_character(self, client_socket):
        self.client_player.name = input("Enter your character's name: ").strip()
        self.client_player.backstory = input("Enter your character's backstory (optional): ").strip()
        client_socket.send(json.dumps({"type": "CHARACTER_SETUP", "name": self.client_player.name, "backstory": self.client_player.backstory}).encode('utf-8'))

if __name__ == "__main__":
    client = Client()
    server_ip = input("Enter server's IP (leave empty if localhost): ")
    if server_ip:
        client.server_ip = server_ip
    client.start_client()