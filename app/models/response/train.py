from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.models.response.snapshot import SnapshotStatusModel


class SmallSnapshotModel(BaseModel):
    """
    Represents a small snapshot of a system or dataset.

    Attributes:
        id (int): The unique identifier for the snapshot.
        name (str): The name of the snapshot.
        status (SnapshotStatusModel): The status object representing the snapshot's current state.
        created (datetime): The creation date and time of the snapshot.
    """
    id: int
    name: str
    status: SnapshotStatusModel
    created: datetime