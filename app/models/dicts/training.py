from dataclasses import dataclass
from torch import Tensor
from typing import Any

@dataclass
class LabelSetDict:
    boxes: Tensor
    labels: Tensor

    def __contains__(self, key: str) -> bool:
        """Enables 'in' keyword to check for attributes."""
        return key in self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        """Allows getting attributes with a default."""
        return getattr(self, key, default)

    def copy(self):
        """Creates a deep copy of the LabelSetDict."""
        return LabelSetDict(self.boxes.clone(), self.labels.clone())