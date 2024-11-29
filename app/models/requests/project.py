from pydantic import BaseModel, Field
from typing import Optional

class CreateProjectRequest(BaseModel):
    """
    Request schema for creating a new project.

    Attributes:
        project_name (str): The title of the project.
        description (Optional[str]): A short description for the project. Optional.
        base_model (str): The CNN model to fine-tune from a list of available models.
    """
    project_name: str = Field(..., title="Project Title", description="The title of the project", examples=['New project 123'])
    description: Optional[str] = Field(None, title="Project Description", description="Create a short description for the project", examples=['Default project description'])
    base_model: str = Field(..., title="CNN Model to Fine-tune", description="Choose a model from the list to start your fine-tuning on", examples=['Single Shot Detection'])

class UpdateProjectRequest(BaseModel):
    """
    Request schema for updating an existing project.

    Attributes:
        project_id (int): The unique identifier of the project to edit.
        project_name (Optional[str]): The new title of the project. Optional.
        description (Optional[str]): The new description of the project. Optional.
    """
    project_id: int = Field(..., title="Project Unique Identifier", description="The unique identifier of the project to edit", examples=[1])
    project_name: Optional[str] = Field(None, title="New Project Title", description="The edited title of the project, optional", examples=['Edited project name'])
    description: Optional[str] = Field(None, title="New Project Description", description="The edited description of the project, optional", examples=['Edited project description'])

class SingleProjectRequest(BaseModel):
    """
    Request schema for deleting a project.

    Attributes:
        project_id (int): The unique identifier of the project to delete.
    """
    project_id: int = Field(..., title="Project Unique Identifier", description="The unique identifier of the project to list the bundles for", examples=[1])
