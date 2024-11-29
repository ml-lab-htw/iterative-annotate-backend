from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form
from typing import List
from playhouse.pool import PooledMySQLDatabase
from app.dependencies import get_db
from app.enums.snapshot_enum import SnapshotStatusMap
from app.models.requests.snapshot import SnapshotListRequest, RemoveSnapshotRequest
from app.models.response.snapshot import SnapshotModel, SnapshotStatusModel, \
    SnapshotLinkedBundlesModel, SnapshotTrainingsInfoModel
from app.services.snapshot_service import SnapshotService

router = APIRouter()

@router.post("/list", response_model=List[SnapshotModel], name='get_all_snapshots')
def get_all_snapshots(
        request: SnapshotListRequest,
        db: PooledMySQLDatabase = Depends(get_db),
):
    """
    Retrieve a list of all snapshots for a given project.

    Args:
        request: The snapshot list request containing the project ID.
        db: The database connection instance.

    Returns:
        A list of SnapshotModel instances representing all snapshots for the project.

    Raises:
        HTTPException: An error occurred while retrieving the snapshots.
    """
    try:
        service = SnapshotService(db=db)
        snapshots = service.get_all_snapshots(project_id=request.project_id)

        return [
            SnapshotModel(
                id=snap.snapshot.id,
                name=snap.snapshot.name,
                path=snap.snapshot.path,
                status=SnapshotStatusModel(
                    code=snap.snapshot.state,
                    value=SnapshotStatusMap.get_string_from_index(snap.snapshot.state)
                ),
                created=snap.snapshot.created_at,
                selected=snap.snapshot.is_selected,
                info=SnapshotTrainingsInfoModel(
                    learning_rate=snap.snapshot.learning_rate,
                    batch_size=snap.snapshot.batch_size,
                    loss=[] if len(snap.snapshot.loss) < 3 else snap.snapshot.loss.split("|")
                ),
                bundles=[
                        SnapshotLinkedBundlesModel(
                            id=bundle.id,
                            created=bundle.uploaded_at,
                            image_count=bundle.images.count()
                        )
                        for bundle in snap.bundles
                ]
            )
            for snap in snapshots
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/delete", response_model=bool, name='remove_snapshot')
def remove_snapshot(
        request: RemoveSnapshotRequest,
        db: PooledMySQLDatabase = Depends(get_db),
):
    """
    Delete a snapshot based on the provided snapshot ID.

    Args:
        request: The remove snapshot request containing the snapshot ID.
        db: The database connection instance.

    Returns:
        A boolean value indicating whether the deletion was successful.

    Raises:
        HTTPException: An error occurred while deleting the snapshot.
    """
    try:
        service = SnapshotService(db)
        service.delete_snapshot(request.snapshot_id)
        return True

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
