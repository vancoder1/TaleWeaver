class Player:
    _id_counter = 0

    def __init__(self, name: str, backstory: str = ""):
        Player._id_counter += 1
        self.player_id = Player._id_counter
        self.name = name
        self.backstory = backstory

    def __str__(self):
        return f"Player {self.player_id}: {self.name}"

    def to_dict(self):
        return {
            "player_id": self.player_id,
            "name": self.name,
            "backstory": self.backstory
        }

    @classmethod
    def from_dict(cls, data):
        player = cls(data["name"], data["backstory"])
        player.player_id = data["player_id"]
        cls._id_counter = max(cls._id_counter, player.player_id)
        return player