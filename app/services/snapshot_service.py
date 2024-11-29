from playhouse.pool import PooledMySQLDatabase

from app.models.database import ProjectORM, ModelSnapshotORM, SnapshotBundleLink, ImageBundleORM
from peewee import DoesNotExist
from typing import List

from app.models.dicts.snapshot import SnapshotListDict
from app.utils.secure_delete import remove_file


class SnapshotService:
    """
    Provides services for managing model snapshots within the application's database.

    This service class encapsulates the logic for retrieving and deleting model snapshots.

    Attributes:
        db: A database connection instance used to perform database transactions.
    """

    def __init__(self, db: PooledMySQLDatabase) -> None:
        """
        Initializes the SnapshotService with a database connection.

        Args:
            db: A database connection instance.
        """
        self.db = db

    def get_all_snapshots(self, project_id: int) -> List[SnapshotListDict]:
        """
        Retrieves all snapshots associated with a given project ID.

        Args:
            project_id (int): The unique identifier of the project.

        Returns:
            list: A list of dictionaries, each containing a snapshot and its linked bundles.

        Raises:
            Exception: If the project with the given ID does not exist or if any other error occurs.
        """
        return_snapshots = []
        with self.db.atomic() as transaction:
            try:
                project = ProjectORM.get_by_id(project_id)
                snapshots = ModelSnapshotORM.select().where(ModelSnapshotORM.project == project)

                for snap in snapshots:
                    linked_bundles = (ImageBundleORM
                                      .select()
                                      .join(SnapshotBundleLink, on=(SnapshotBundleLink.bundle == ImageBundleORM.id))
                                      .where(SnapshotBundleLink.snapshot == snap))

                    return_snapshots.append(SnapshotListDict(
                        snapshot=snap,
                        bundles=linked_bundles
                    ))

                return return_snapshots

            except DoesNotExist:
                raise Exception(f"Project with ID {project_id} does not exist.")
            except Exception as e:
                raise e

    def delete_snapshot(self, snapshot_id: int) -> None:
        """
        Deletes a snapshot and its associated data from the database based on the snapshot ID.

        Args:
            snapshot_id (int): The unique identifier of the snapshot to delete.

        Raises:
            Exception: If the snapshot with the given ID does not exist or if any other error occurs.
        """
        with self.db.atomic() as transaction:
            try:
                snapshot = ModelSnapshotORM.get_by_id(snapshot_id)

                # Delete all related SnapshotBundleLink entries
                for link in snapshot.bundle_link:
                    link.delete_instance()

                # Remove snapshot file
                label_set_file = snapshot.path.replace('.pth', '_labels.json')
                remove_file(snapshot.path)
                remove_file(label_set_file)

                snapshot.delete_instance()

            except DoesNotExist:
                raise Exception(f"Snapshot with ID {snapshot_id} does not exist.")
            except Exception as e:
                transaction.rollback()
                raise e
