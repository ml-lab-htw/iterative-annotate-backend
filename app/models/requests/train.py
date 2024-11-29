from pydantic import BaseModel, Field
from typing import List, Optional

class TrainingRequest(BaseModel):
    """
    Request schema for initiating a training process.

    Attributes:
        snapshot_name (str): An identifying name for the new model snapshot.
        bundle_ids (List[int]): A list of bundle IDs to use for training the model.
        base_snapshot_id (Optional[int]): The ID of the base snapshot to use, if any.
        batch_size (int): The training batch size per epoch. This is an advanced setting.
        epochs (int): The number of training epochs. This is an advanced setting.
        learning_rate (float): The learning rate for training. This is an advanced setting.
    """
    snapshot_name: str = Field(..., title="Snapshot Name", description="Give an identifying name to the new model snapshot", example="New snapshot")
    bundle_ids: List[int] = Field(..., title="Bundle ID List", description="List of bundle IDs to use for training the model", example=[1, 2])
    base_snapshot_id: Optional[int] = Field(default=None, title="Base Snapshot ID", description="ID of the base snapshot to use, if any", example='null')
    batch_size: int = Field(default=50, title="Batch Size", description="Define the training batch size per epoch (advanced)", example=50)
    epochs: int = Field(default=20, title="Epochs", description="Define the number of training epochs (advanced)", example=20)
    learning_rate: float = Field(default=0.01, title="Learning Rate", description="Define the learning rate (advanced)", example=0.01)