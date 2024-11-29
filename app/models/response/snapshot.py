from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class SnapshotLinkedBundlesModel(BaseModel):
    """
    Represents the linked bundles associated with a snapshot.

    Attributes:
        id (int): The unique identifier for the linked bundle.
        created (datetime): The creation date and time of the linked bundle.
        image_count (int): The number of images contained in the linked bundle.
    """
    id: int
    created: datetime
    image_count: int


class SnapshotStatusModel(BaseModel):
    """
    Represents the status of a snapshot.

    Attributes:
        value (str): The descriptive status of the snapshot.
        code (int): The numerical code representing the status of the snapshot.
    """
    value: str
    code: int

class SnapshotTrainingsInfoModel(BaseModel):
    learning_rate: float
    batch_size: int
    loss: List[float]


class SnapshotModel(BaseModel):
    """
    Represents a snapshot of a system or dataset.

    Attributes:
        id (int): The unique identifier for the snapshot.
        name (str): The name of the snapshot.
        path (str): The file system path where the snapshot is stored.
        status (SnapshotStatusModel): The status object representing the snapshot's current state.
        selected (bool): A flag indicating whether the snapshot is selected.
        created (datetime): The creation date and time of the snapshot.
        bundles (List[SnapshotLinkedBundlesModel]): A list of linked bundles associated with the snapshot.
    """
    id: int
    name: str
    path: str
    status: SnapshotStatusModel
    info: SnapshotTrainingsInfoModel
    selected: bool
    created: datetime
    bundles: List[SnapshotLinkedBundlesModel] = []
