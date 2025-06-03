from typing import Dict, Any

class Player:
    _id_counter = 0

    def __init__(self, name: str, backstory: str = ""):
        Player._id_counter += 1
        self.player_id: int = Player._id_counter
        self.name: str = name
        self.backstory: str = backstory

    def __str__(self) -> str:
        return f"Player {self.player_id}: {self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "backstory": self.backstory
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        player = cls(data["name"], data["backstory"])
        player.player_id = data["player_id"]
        cls._id_counter = max(cls._id_counter, player.player_id)
        return player