import enum
import os

from dotenv import load_dotenv

from app.models.database import ImageEntryORM

"""
Utility functions for handling image file operations within the application.

This module provides utility functions that are used to interact with image files,
such as retrieving the path to an image file based on its version (original, thumbnail, or transformed),
and loading environment variables related to image file handling.

Classes:
    IMAGE_VERSION: An enumeration that defines constants for different image versions.

Functions:
    get_image_path(image: ImageEntryORM, version: IMAGE_VERSION, local_path: bool) -> str:
        Constructs and returns the path to an image file based on the specified version and whether
        the path should be local or server-based.
"""

load_dotenv()

class IMAGE_VERSION(enum.IntEnum):
    ORIGINAL = 1
    THUMBNAIL = 2
    TRANSFORMED = 3


# Load environment variables from .env file
load_dotenv()
# Access environment variable for SERVER DOMAIN
SERVER_DOMAIN = os.getenv("SERVER_DOMAIN", "http://localhost:8000")


def get_image_path(image: ImageEntryORM, version: IMAGE_VERSION, local_path = False) -> str:
    match version:
        case IMAGE_VERSION.ORIGINAL:
            folder_by_type = "original"
        case IMAGE_VERSION.THUMBNAIL:
            folder_by_type = "thumbnail"
        case IMAGE_VERSION.TRANSFORMED:
            folder_by_type = "transformed"
        case _:
            raise Exception("Invalid file version")

    if local_path:
        base_dir = os.getenv('BASE_DIR')
        return os.path.join(base_dir, image.path, folder_by_type, image.filename)
    else:
        return f"{SERVER_DOMAIN}/{image.path}{folder_by_type}/{image.filename}"

