class Player:
    _id_counter = 0

    def __init__(self, name="", backstory=""):
        Player._id_counter += 1
        self.player_id = Player._id_counter
        self.name = name
        self.backstory = backstory

    def __str__(self):
        return f"Player {self.player_id}: {self.name}"