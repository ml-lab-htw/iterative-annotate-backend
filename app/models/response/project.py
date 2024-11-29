from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProjectInfoModel(BaseModel):
    """
    Schema for project information.

    Attributes:
        bundle_count (int): The number of bundles associated with the project.
        image_count (int): The number of images associated with the project.
        snapshot_count (int): The number of snapshots associated with the project.
    """
    bundle_count: int
    image_count: int
    snapshot_count: int

class ProjectModel(BaseModel):
    """
    Schema for a project.

    Attributes:
        id (int): The unique identifier of the project.
        name (str): The name of the project.
        description (Optional[str]): An optional description of the project.
        model (str): The model associated with the project.
        created (datetime): The date and time when the project was created.
        info (Optional[ProjectInfoModel]): Optional detailed information about the project.
    """
    id: int
    name: str
    description: Optional[str] = None
    model: str
    created: datetime
    info: Optional[ProjectInfoModel] = None