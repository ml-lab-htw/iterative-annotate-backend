from typing import List
from playhouse.pool import PooledMySQLDatabase
from fastapi import APIRouter, Depends, HTTPException, Form
from app.dependencies import get_db
from app.models.requests.project import CreateProjectRequest, UpdateProjectRequest, SingleProjectRequest
from app.models.response.bundle import ImageBundleModel, SmallImageModel, BundleStatusModel
from app.models.response.project import ProjectModel, ProjectInfoModel

from app.utils.image_files import IMAGE_VERSION, get_image_path
from app.utils.date_util import format_date

from app.enums.model_enum import BaseModelString, BaseModelMap
from app.enums.status_enum import BundleStatusMap

from app.services.project_service import ProjectService


router = APIRouter()


@router.get("/list",
            response_model=List[ProjectModel],
            name='project_list',
            description='Get list of all project')
def project_list(db: PooledMySQLDatabase = Depends(get_db)):
    """
    Retrieve a list of all projects from the database.

    Args:
        db (PooledMySQLDatabase): The database connection instance.

    Returns:
        List[ProjectModel]: A list of project models containing project details.

    Raises:
        HTTPException: An error occurred while retrieving the project list.
    """
    try:
        service = ProjectService(db)
        projects_data = service.project_list()

        return [
            ProjectModel(
                id=project.orm.id,
                name=project.orm.name,
                description=project.orm.description,
                model=BaseModelMap.get_string_from_index(project.orm.base_model),
                created=project.orm.created_at,
                info=ProjectInfoModel(
                    bundle_count=project.bundle_cnt,
                    image_count=project.image_sum,
                    snapshot_count=project.snapshot_cnt
                )
            )
            for project in projects_data
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/info",
             response_model=ProjectModel,
             name='project_info',
             description='Get information about a single project')
def project_info(
        request: SingleProjectRequest,
        db: PooledMySQLDatabase = Depends(get_db)):

    try:
        service = ProjectService(db)
        project = service.project_info(request.project_id)

        return ProjectModel(
            id=project.orm.id,
            name=project.orm.name,
            description=project.orm.description,
            model=BaseModelMap.get_string_from_index(project.orm.base_model),
            created=project.orm.created_at,
            info=ProjectInfoModel(
                bundle_count=project.bundle_cnt,
                image_count=project.image_sum,
                snapshot_count=project.snapshot_cnt
            )
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/create",
             response_model=ProjectModel,
             name='create_project',
             description='Create a new project')
def create_project(
        request: CreateProjectRequest,
        db: PooledMySQLDatabase = Depends(get_db)):
    """
    Create a new project with the given details.

    Args:
        request (CreateProjectRequest): The project creation request containing the project name, description, and base model.
        db (PooledMySQLDatabase): The database connection instance.

    Returns:
        ProjectModel: The created project model with project details.

    Raises:
        HTTPException: An error occurred while creating the project.
    """
    try:
        service = ProjectService(db)

        base_model_enum = BaseModelMap.get_enum_from_string(request.base_model)
        if request.description is None:
            description = 'No description provided'

        project = service.create_project(request.project_name, request.description, base_model_enum)

        return ProjectModel(
                id=project.id,
                name=project.name,
                description=project.description,
                model=request.base_model,
                created=project.created_at
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/edit",
             response_model=bool,
             name='edit_project',
             description='Edit a selected project')
def update_project(
        request: UpdateProjectRequest,
        db: PooledMySQLDatabase = Depends(get_db)):
    """
    Update the details of an existing project.

    Args:
        request (UpdateProjectRequest): The project update request containing the project ID, new name, and new description.
        db (PooledMySQLDatabase): The database connection instance.

    Returns:
        bool: True if the update was successful, False otherwise.

    Raises:
        HTTPException: An error occurred while updating the project.
    """
    try:
        service = ProjectService(db)
        service.update_project(request.project_id, request.project_name, request.description)

        return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/delete",
             response_model=bool,
             name='delete_project',
             description="Delete a selected project and all it's contents")
def delete_project(
        request: SingleProjectRequest,
        db: PooledMySQLDatabase = Depends(get_db)):
    """
    Delete a project and all its associated contents from the database.

    Args:
        request (ProjectDeleteRequest): The project deletion request containing the project ID.
        db (PooledMySQLDatabase): The database connection instance.

    Returns:
        bool: True if the deletion was successful, False otherwise.

    Raises:
        HTTPException: An error occurred while deleting the project.
    """
    try:
        service = ProjectService(db)
        service.delete_project(request.project_id)

        return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
