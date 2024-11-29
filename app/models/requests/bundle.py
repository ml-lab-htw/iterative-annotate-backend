from typing import List

from pydantic import BaseModel, Field

class BundleListRequest(BaseModel):
    """
    Request schema for listing bundles associated with a project.

    Attributes:
        project_id (int): The unique identifier of the project to list the bundles for.
    """
    project_id: int = Field(..., title="Project Unique Identifier", description="The unique identifier of the project to list the bundles for", example=1)

class RemoveBundleRequest(BaseModel):
    """
    Request schema for removing a specific image bundle.

    Attributes:
        bundle_id (int): The unique identifier of the bundle of images.
    """
    bundle_id: int = Field(..., title="Image Bundle Unique Identifier", description="The unique identifier of the bundle of images", example=3)

class GetAnnotatedBundleRequest(BaseModel):
    """
    Request schema for retrieving an annotated bundle of images.

    Attributes:
        bundle_id (int): The unique identifier of the bundle of images to be retrieved.
    """
    bundle_id: int = Field(..., title="Bundle Unique Identifier", description="The unique identifier of the bundle of images to be retrieved.", example=3)

class AnnotationsReviewBox(BaseModel):
    """
    Schema for the bounding box of an annotation in a review.

    Attributes:
        x (float): The X coordinate of the annotation box's top-left corner.
        y (float): The Y coordinate of the annotation box's top-left corner.
        width (float): The width of the annotation box.
        height (float): The height of the annotation box.
    """
    x: int = Field(..., title="X Coordinate", description="The X coordinate of the annotation box's top-left corner.", example=120.5)
    y: int = Field(..., title="Y Coordinate", description="The Y coordinate of the annotation box's top-left corner.", example=75.0)
    width: int = Field(..., title="Width", description="The width of the annotation box.", example=250.0)
    height: int = Field(..., title="Height", description="The height of the annotation box.", example=160.5)

class AnnotationsReviewEntry(BaseModel):
    """
    Schema for an entry in the annotations review.

    Attributes:
        id (int): The unique identifier of the annotation being reviewed.
        label (str): The label assigned to the annotation.
        box (AnnotationsReviewBox): The bounding box of the annotation.
    """
    id: int = Field(..., title="Annotation ID", description="The unique identifier of the annotation being reviewed.", example=21)
    label: str = Field(..., title="Label", description="The label assigned to the annotation.", example="cat")
    box: AnnotationsReviewBox = Field(..., title="Annotation Box", description="The bounding box of the annotation.")
    edited: bool = Field(..., title="Annotation edited", description="Displays if the annotation was edited by the user")
    active: bool = Field(..., title="Annotation active", description="Inactive annotations are neglected during training")

class ReviewAnnotationRequest(BaseModel):
    """
    Request schema for reviewing annotations of a specific image.

    Attributes:
        image_id (int): The unique identifier of the image for which annotations are being reviewed.
        reviewed_annotations (List[AnnotationsReviewEntry]): A list of annotations that have been reviewed for this image.
    """
    image_id: int = Field(..., title="Image Unique Identifier", description="The unique identifier of the image for which annotations are being reviewed.", example=6)
    reviewed_annotations: List[AnnotationsReviewEntry] = Field(..., title="Reviewed Annotations", description="A list of annotations that have been reviewed for this image.")

class TestBundleInfoData(BaseModel):
    box_created_cnt: int = Field(...)
    box_removed_cnt: int = Field(...)
    box_moved_cnt: int = Field(...)
    label_updated_cnt: int = Field(...)
    navigated_img_cnt: int = Field(...)

class TestBundleRequest(BaseModel):
    bundle_id: int = Field(..., title="Bundle Unique Identifier")
    duration: int = Field(..., title="Seconds of editing session")
    data: TestBundleInfoData = Field(..., title="Labeling behavior data")
