from dataclasses import dataclass
from typing import List


@dataclass
class InferenceResultDict:
    label: str
    label_id: int
    box: List[float]
    score: float

