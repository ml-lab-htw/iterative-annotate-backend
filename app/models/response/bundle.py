from datetime import datetime

from pydantic import BaseModel
from typing import List


class AnnotationBox(BaseModel):
    """
    Represents a bounding box for an annotation on an image.

    Attributes:
        x (float): The X coordinate of the top-left corner of the box.
        y (float): The Y coordinate of the top-left corner of the box.
        width (float): The width of the box.
        height (float): The height of the box.
    """
    x: int
    y: int
    width: int
    height: int

class AnnotationModel(BaseModel):
    """
    Represents an annotation on an image.

    Attributes:
        id (int): The unique identifier for the annotation.
        label (str): The label of the annotation.
        score (float): The confidence score of the annotation.
        active (bool): The status indicating if the annotation is active.
        box (AnnotationBox): The bounding box of the annotation.
    """
    id: int
    label: str
    score: float
    active: bool
    edited: bool
    box: AnnotationBox

class SmallImageModel(BaseModel):
    """
    Represents a small image, typically a thumbnail.

    Attributes:
        id (int): The unique identifier for the image.
        thump (str): The URL or path to the thumbnail image.
    """
    id: int
    thump: str

class BundleStatusModel(BaseModel):
    """
    Represents the status of an image bundle.

    Attributes:
        value (str): The descriptive status of the bundle.
        code (int): The numerical code representing the status.
    """
    value: str
    code: int

class ImageBundleModel(BaseModel):
    """
    Represents a collection of images as a bundle.

    Attributes:
        id (int): The unique identifier for the image bundle.
        images (List[SmallImageModel]): A list of small images associated with the bundle.
        uploaded (datetime): The date and time when the bundle was uploaded.
        status (BundleStatusModel): The status of the image bundle.
    """
    id: int
    images: List[SmallImageModel] = []
    uploaded: datetime
    status: BundleStatusModel

class ImagePathModel(BaseModel):
    """
    Represents the paths to different versions of an image.

    Attributes:
        thumbnail (str): The path to the thumbnail version of the image.
        transformed (str): The path to the transformed version of the image.
        original (str): The path to the original version of the image.
    """
    thumbnail: str
    transformed: str
    original: str

class AnnotatedImageModel(BaseModel):
    """
    Represents an image with annotations.

    Attributes:
        id (int): The unique identifier for the annotated image.
        paths (ImagePathModel): The paths to different versions of the image.
        annotations (List[AnnotationModel]): A list of annotations on the image.
    """
    id: int
    paths: ImagePathModel
    annotations: List[AnnotationModel] = []