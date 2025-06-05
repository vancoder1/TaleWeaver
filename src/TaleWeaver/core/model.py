from typing import Dict, Any

class Player:
    def __init__(self, name: str = "Adventurer", backstory: str = "A mysterious wanderer."):
        self.name: str = name or "Adventurer"
        self.backstory: str = backstory or "A mysterious wanderer."
    def to_dict(self) -> Dict[str, Any]: return {"name": self.name, "backstory": self.backstory}
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        return cls(name=data.get("name"), backstory=data.get("backstory"))