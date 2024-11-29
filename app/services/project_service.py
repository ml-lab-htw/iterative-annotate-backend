from pathlib import Path
from typing import List, Dict, Optional

from app.models.database import ProjectORM, ModelSnapshotORM
from peewee import DoesNotExist
from playhouse.pool import PooledMySQLDatabase

from app.enums.model_enum import BaseModelEnum
from app.models.dicts.project import ProjectDataDict
from app.utils.image_files import get_image_path, IMAGE_VERSION
from app.utils.secure_delete import remove_file, remove_folder


class ProjectService:
    """
    Provides services for managing projects within the application's database.

    This service class encapsulates the logic for creating, listing, updating, and deleting projects,
    as well as handling associated data such as image bundles and model snapshots.

    Attributes:
        db: A database connection instance used to perform database transactions.
    """

    def __init__(self, db: PooledMySQLDatabase) -> None:
        """
        Initializes the ProjectService with a database connection.

        Args:
            db: A database connection instance.
        """
        self.db = db

    def project_list(self) -> List[ProjectDataDict]:
        """
        Retrieves a list of all projects with their associated data.

        Returns:
            A list of dictionaries, each containing data about a project, including the number of
            image bundles, the total number of images, and the count of model snapshots.

        Raises:
            Exception: If any error occurs during the database transaction.
        """
        with self.db.atomic() as transaction:
            try:
                # Create a new project instance and save it to the database
                project_data = []  # List to hold information about each project

                all_projects = ProjectORM.select()
                for prj in all_projects:
                    bundles_list = prj.bundles
                    bundle_count = bundles_list.count()

                    image_sum = 0  # Initialize sum of images for this project
                    for bundle in bundles_list:
                        image_sum += bundle.images.count()  # Summing the number of images in each bundle

                    snapshot_count = ModelSnapshotORM.select().where(ModelSnapshotORM.project == prj).count()

                    project_data.append(ProjectDataDict(
                        orm=prj,
                        bundle_cnt=bundle_count,
                        image_sum=image_sum,
                        snapshot_cnt=snapshot_count
                    ))
                return project_data

            except Exception as e:
                transaction.rollback()  # Roll back the transaction on error
                raise e

    def project_info(self, project_id):
        with self.db.atomic() as transaction:
            try:
                project = ProjectORM.get_by_id(project_id)

                bundles_list = project.bundles
                bundle_count = bundles_list.count()

                image_sum = 0  # Initialize sum of images for this project
                for bundle in bundles_list:
                    image_sum += bundle.images.count()  # Summing the number of images in each bundle

                snapshot_count = ModelSnapshotORM.select().where(ModelSnapshotORM.project == project).count()

                return ProjectDataDict(
                    orm=project,
                    bundle_cnt=bundle_count,
                    image_sum=image_sum,
                    snapshot_cnt=snapshot_count
                )

            except Exception as e:
                transaction.rollback()  # Roll back the transaction on error
                raise e


    def create_project(self, _name: str, _description: str, _base_model: BaseModelEnum) -> ProjectORM:
        """
        Creates a new project in the database.

        Args:
            _name: The name of the project.
            _description: A description of the project.
            _base_model: The base model enumeration value for the project.

        Returns:
            The newly created ProjectORM instance.

        Raises:
            Exception: If any error occurs during the database transaction.
        """
        with self.db.atomic() as transaction:
            try:
                return ProjectORM.create(
                    name=_name,
                    description=_description,
                    base_model=_base_model
                )

            except Exception as e:
                transaction.rollback()  # Roll back the transaction on error
                raise e

    def update_project(self, _id: int, new_name: str, new_description: str) -> None:
        """
        Updates the specified project's name and/or description.

        Args:
            _id: The ID of the project to update.
            new_name: The new name for the project, if provided.
            new_description: The new description for the project, if provided.

        Raises:
            Exception: If the project with the given ID does not exist or any other error occurs during the database transaction.
        """
        with self.db.atomic() as transaction:
            try:
                project = ProjectORM.get_by_id(_id)

                change_flag = False

                if new_name is not None:
                    project.name = new_name
                    change_flag = True

                if new_description is not None:
                    project.description = new_description
                    change_flag = True

                if change_flag:
                    # Save the changes to the database
                    project.save()

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Project with ID {_id} does not exist.")
            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e

    def delete_project(self, _id: int) -> None:
        """
        Deletes the specified project and all associated data from the database.

        Args:
            _id: The ID of the project to delete.

        Raises:
            Exception: If the project with the given ID does not exist or any other error occurs during the database transaction.
        """
        with self.db.atomic() as transaction:
            try:
                project = ProjectORM.get_by_id(_id)
                # Delete linked objects from SnapshotBundleLink before deleting snapshots or bundles
                # As snapshots reference bundles, links must be cleared first
                for snapshot in project.snapshots:
                    for link in snapshot.bundle_link:
                        link.delete_instance()

                # Now safe to delete snapshots
                for snapshot in project.snapshots:
                    snapshot.delete_instance()

                bundle_folder = None

                for bundle in project.bundles:
                    for image in bundle.images:
                        bundle_folder = image.path # Store the path to check the parent folder later

                        for annotation in image.annotations:
                            annotation.delete_instance()

                        image.delete_instance()

                    bundle.delete_instance()
                # After all related objects have been deleted, delete the project itself
                project.delete_instance()

                if bundle_folder is not None:
                    project_folder = Path(bundle_folder).parent
                    remove_folder(str(project_folder))

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Project with ID {_id} does not exist.")
            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e
