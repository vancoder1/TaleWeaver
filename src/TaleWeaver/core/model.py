from typing import Dict, Any

class Player:
    def __init__(self, name: str = "Adventurer", backstory: str = "A mysterious wanderer with an unknown past."):
        self.name: str = name if name else "Adventurer" # Ensure name is not empty
        self.backstory: str = backstory if backstory else "A mysterious wanderer with an unknown past."

    def __str__(self) -> str:
        return f"Character: {self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "backstory": self.backstory}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        # Uses .get() to provide defaults if keys are missing or values are None/empty
        return cls(
            name=data.get("name") or "Adventurer",
            backstory=data.get("backstory") or "A mysterious wanderer with an unknown past."
        )