from typing import List

from pydantic import BaseModel, Field

class SnapshotListRequest(BaseModel):
    """
    Request schema for listing snapshots associated with a project.

    Attributes:
        project_id (int): The unique identifier of the project to list the snapshots for.
    """
    project_id: int = Field(..., title="Project Unique Identifier", description="The unique identifier of the project to list the snapshots for", example=1)

class RemoveSnapshotRequest(BaseModel):
    """
    Request schema for removing a specific snapshot.

    Attributes:
        snapshot_id (int): The unique identifier of the snapshot to delete.
    """
    snapshot_id: int = Field(..., title="Snapshot Unique Identifier", description="The unique identifier of the snapshot to delete", example=3)
