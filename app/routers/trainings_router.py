from typing import List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form
from playhouse.pool import PooledMySQLDatabase
from app.dependencies import get_db
from app.enums.snapshot_enum import SnapshotStatusMap
from app.models.requests.train import TrainingRequest
from app.models.response.snapshot import SnapshotStatusModel
from app.models.response.train import SmallSnapshotModel
from app.services.trainings_service import TrainingsService

router = APIRouter()


@router.post("/bundles",
            response_model=SmallSnapshotModel,
            name='model_finetune',
            description='Finetune a model from bundles of images')
async def train_model(
        background_tasks: BackgroundTasks,
        request: TrainingRequest,
        db: PooledMySQLDatabase = Depends(get_db),
):
    """
    Starts the training of a model using the provided training request data.

    This endpoint initiates a new training snapshot and begins the model training process
    as a background task. It returns a small snapshot model containing the details of the
    initiated training snapshot.

    Args:
        background_tasks (BackgroundTasks): The background tasks manager to use for running
                                            the training process in the background.
        request (TrainingRequest): The training request containing the parameters for training
                                   such as batch size, epochs, learning rate, etc.
        db (PooledMySQLDatabase): The database connection instance.

    Returns:
        SmallSnapshotModel: An instance containing the details of the initiated training snapshot.

    Raises:
        HTTPException: An error occurred during the training initiation process.
    """
    service = None
    try:
        service = TrainingsService(db=db, batch_size=request.batch_size, epochs=request.epochs, learning_rate=request.learning_rate)
        snapshot_orm = service.start_snapshot(snapshot_name=request.snapshot_name, base_snapshot_id=request.base_snapshot_id, bundle_id_array=request.bundle_ids)
        background_tasks.add_task(service.train_model)

        return SmallSnapshotModel(
            id=snapshot_orm.id,
            name=snapshot_orm.name,
            status=SnapshotStatusModel(
                code=snapshot_orm.state,
                value=SnapshotStatusMap.get_string_from_index(snapshot_orm.state)
            ),
            created=snapshot_orm.created_at
        )

    except Exception as e:
        if service:
            service.abort_snapshot()

        raise HTTPException(status_code=400, detail=str(e))
