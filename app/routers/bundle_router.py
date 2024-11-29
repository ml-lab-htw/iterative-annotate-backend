from typing import List
from xmlrpc.client import Boolean

from app.dependencies import get_db
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Form, Depends

from app.models.requests.bundle import BundleListRequest, RemoveBundleRequest, GetAnnotatedBundleRequest, \
    ReviewAnnotationRequest, AnnotationsReviewEntry, AnnotationsReviewBox, TestBundleRequest
from app.models.response.bundle import ImagePathModel, AnnotationModel, AnnotationBox, AnnotatedImageModel, \
    ImageBundleModel, SmallImageModel, BundleStatusModel
from app.services.bundle_service import BundleService
from playhouse.pool import PooledMySQLDatabase

from app.utils.image_files import get_image_path, IMAGE_VERSION
from app.enums.status_enum import BundleStatusMap

router = APIRouter()


@router.post("/list",
             response_model=List[ImageBundleModel],
             name='list_bundles',
             description='List all image bundles for a selected project')
def list_bundles(
        request: BundleListRequest,
        db: PooledMySQLDatabase = Depends(get_db)):
    """
    List all image bundles for a selected project.

    Args:
        request: The bundle list request containing the project ID.
        db: The database connection instance.

    Returns:
        A list of ImageBundleModel instances representing all image bundles for the project.

    Raises:
        HTTPException: An error occurred while listing the bundles.
    """
    try:
        service = BundleService(db)
        bundle_orm = service.list_bundles(request.project_id)

        return [
            ImageBundleModel(
                id=bundle.id,
                images=[
                    SmallImageModel(
                        id=image.id,
                        thump=get_image_path(image,IMAGE_VERSION.THUMBNAIL),
                    )
                    for image in bundle.images
                ],
                uploaded=bundle.uploaded_at,
                status=BundleStatusModel(
                    code=bundle.status,
                    value=BundleStatusMap.get_string_from_index(bundle.status)
                )
            )
            for bundle in bundle_orm
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/delete",
             response_model=bool,
             name='remove_bundle',
             description='Remove a bundle of images and all its data from a given project')
def remove_bundle(
        request: RemoveBundleRequest,
        db: PooledMySQLDatabase = Depends(get_db)):
    """
    Remove a bundle of images and all its data from a given project.

    Args:
        request: The remove bundle request containing the bundle ID.
        db: The database connection instance.

    Returns:
        A boolean indicating whether the bundle was successfully removed.

    Raises:
        HTTPException: An error occurred while removing the bundle.
    """
    try:
        service = BundleService(db)
        service.remove_bundle(request.bundle_id)

        return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/images/upload",
             response_model=ImageBundleModel,
             name="upload_bundle",
             description="Upload a bundle of images to a project. \
             After the Upload, all images will be automatically annotated and processed for predictions.")
async def upload_images(
        background_tasks: BackgroundTasks,
        project_id: int = Form(...,
                               title='Project Unique Identifier',
                               description="The unique identifier of the project to upload images to",
                               example=1
                               ),
        files: List[UploadFile] = File(...,
                                       title='Image Files',
                                       description="A list of image files to upload to the project"),
        base_snapshot_id: int = Form(default=None,
                                     title="Base Snapshot ID",
                                     description="ID of the base snapshot to use, if any", example=None),
        inference_confidence_threshold: float = Form(default=None,
                                     title="Confidence threshold for inference",
                                     description="The confidence level from 0 to 1, at which annotations are included in the inference results", example=0.75),
        db: PooledMySQLDatabase = Depends(get_db)
):
    """
    Upload a bundle of images to a project and start automatic annotation and processing for predictions.

    Args:
        background_tasks: The background tasks manager to use for running long-running operations.
        project_id: The unique identifier of the project to upload images to.
        files: A list of image files to upload to the project.
        base_snapshot_id: ID of the base snapshot to use, if any.
        db: The database connection instance.

    Returns:
        An ImageBundleModel instance representing the uploaded image bundle.

    Raises:
        HTTPException: An error occurred during the upload process.
    """
    try:
        print(f"\nthreshold: {inference_confidence_threshold}\n")
        service = BundleService(db=db, confidence_threshold=inference_confidence_threshold)
        bundle_object = await service.create_new_bundle(project_id)

        image_list = await service.upload_and_process(files, project_id, bundle_object)
        background_tasks.add_task(service.start_bundle_inference, image_list, bundle_object, base_snapshot_id)

        return ImageBundleModel(
            id=bundle_object.id,
            images=[
                SmallImageModel(
                    id=image.id,
                    thump=get_image_path(image, IMAGE_VERSION.THUMBNAIL)
                )
                for image in bundle_object.images
            ],
            uploaded=bundle_object.uploaded_at,
            status=BundleStatusModel(
                code=bundle_object.status,
                value=BundleStatusMap.get_string_from_index(bundle_object.status)
            )
        )

    except Exception as e:
        # TODO: remove already uploaded images
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotation/get",
             response_model=List[AnnotatedImageModel],
             name="annotated_bundle",
             description="Get a bundle of annotated images with all necessary bounding boxes and labels.")
def get_annotated_bundle(
    request: GetAnnotatedBundleRequest,
    db: PooledMySQLDatabase = Depends(get_db)
):
    """
    Get a bundle of annotated images with all necessary bounding boxes and labels.

    Args:
        request: The get annotated bundle request containing the bundle ID.
        db: The database connection instance.

    Returns:
        A list of AnnotatedImageModel instances representing the annotated images.

    Raises:
        HTTPException: An error occurred while retrieving the annotated images.
    """
    try:
        service = BundleService(db)
        image_list = service.get_annotated_bundle(request.bundle_id)

        return [
            AnnotatedImageModel(
                id=image.id,
                paths=ImagePathModel(
                    original=get_image_path(image, IMAGE_VERSION.ORIGINAL),
                    thumbnail=get_image_path(image, IMAGE_VERSION.THUMBNAIL),
                    transformed=get_image_path(image, IMAGE_VERSION.TRANSFORMED)
                ),
                annotations=[
                    AnnotationModel(
                        id=annotation.id,
                        label=annotation.label,
                        score=annotation.score,
                        active=annotation.active,
                        edited=False if annotation.original_state else True,
                        box=AnnotationBox(
                            x=annotation.x,
                            y=annotation.y,
                            width=annotation.width,
                            height=annotation.height
                        )
                    )
                    for annotation in image.annotations if annotation.active
                ]
            )
            for image in image_list
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotation/review",
             response_model=ReviewAnnotationRequest,
             name="review_annotation",
             description="Update a single annotation in the database for a selected image")
def review_annotation(
    request: ReviewAnnotationRequest,
    db: PooledMySQLDatabase = Depends(get_db)
):
    """
    Update al annotation in the database for a single selected image.

    Args:
        request: The review annotation request containing the image ID and the reviewed annotations.
        db: The database connection instance.

    Returns:
        A boolean indicating whether the annotation was successfully updated.

    Raises:
        HTTPException: An error occurred while updating the annotation.
    """
    try:
        service = BundleService(db)
        orm_annotation_list = service.update_annotation(request.image_id, request.reviewed_annotations)

        return ReviewAnnotationRequest(
            image_id=request.image_id,
            reviewed_annotations=[
                AnnotationsReviewEntry(
                    id=annotation.id,
                    label=annotation.label,
                    active=annotation.active,
                    edited=False if annotation.original_state else True,
                    box=AnnotationsReviewBox(
                        x=annotation.x,
                        y=annotation.y,
                        width=annotation.width,
                        height=annotation.height
                    )
                )
                for annotation in orm_annotation_list
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/test",
             response_model=bool,
             name="test_labeling",
             description="Get data about the manual labeling process")
def submit_test(
    request: TestBundleRequest,
    db: PooledMySQLDatabase = Depends(get_db)
):
    """
    Update all annotation in the database for a single selected image.

    Args:
        request: The review annotation request containing the image ID and the reviewed annotations.
        db: The database connection instance.

    Returns:
        A boolean indicating whether the annotation was successfully updated.

    Raises:
        HTTPException: An error occurred while updating the annotation.
    """
    try:
        service = BundleService(db)
        service.submit_test(request.bundle_id, request.duration, request.data)

        return True

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))