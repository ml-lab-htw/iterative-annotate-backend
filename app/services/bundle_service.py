import os
from datetime import datetime

import torch
import aiofiles
import asyncio
import json
import time

from dotenv import load_dotenv
from fastapi import UploadFile
from peewee import DoesNotExist
from playhouse.pool import PooledMySQLDatabase

from PIL import Image, ExifTags
import piexif
from typing import List, Dict, Optional, Tuple


from app.enums.snapshot_enum import SnapshotStatusEnum
from app.models.database import ImageBundleORM, ImageEntryORM, ProjectORM, AnnotationORM, ModelSnapshotORM
from app.models.dicts.bundle import InferenceResultDict
from app.models.requests.bundle import AnnotationsReviewEntry, TestBundleInfoData
from app.utils.custom_dataset import COCO_INSTANCE_CATEGORY_NAMES
from app.utils.custom_model import CustomModel
from app.utils.filename import generate_random_filename
from app.utils.image_files import get_image_path, IMAGE_VERSION
from app.enums.status_enum import BundleStatusEnum
from app.utils.secure_delete import remove_folder
from app.utils.transformer import Transformer
from app.utils import colors

class BundleService:
    """
    Service class for handling operations related to image bundles.

    Attributes:
        db (PooledMySQLDatabase): The database connection instance.
        ssd_model (CustomModel, optional): The SSD model for image processing, initialized as None.
        device (torch.device, optional): The device on which the model will run, initialized as None.
        label_set (list): A list of category names for image labeling.
        min_confidence (float): The minimum confidence threshold for image processing.
    """

    def __init__(self, db: PooledMySQLDatabase, confidence_threshold: float = 0.75, result_threshold: int = 100) -> None:
        """
        Initializes the BundleService with a database connection and a confidence threshold.

        Args:
            db (PooledMySQLDatabase): The database connection instance.
            confidence_threshold (float, optional): The minimum confidence threshold for image processing.
        """
        self.db = db
        self.ssd_model = None
        self.label_set = COCO_INSTANCE_CATEGORY_NAMES.copy()
        self.min_confidence = confidence_threshold
        self.max_results = result_threshold
        load_dotenv()
        self.base_dir = os.getenv('BASE_DIR')

    async def create_new_bundle(self, project_id: int) -> ImageBundleORM:
        """
        Creates a new image bundle associated with a given project.

        Args:
            project_id (int): The unique identifier of the project.

        Returns:
            ImageBundleORM: The newly created image bundle ORM instance.

        Raises:
            Exception: If the project with the given ID does not exist or any other error occurs during the creation of the image bundle.
        """
        with self.db.atomic() as transaction:
            try:
                # Create a new project instance and save it to the database
                project = ProjectORM.get_by_id(project_id)
                return ImageBundleORM.create(project=project)

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Project with ID {project_id} does not exist.")
            except Exception as e:
                transaction.rollback()  # Roll back the transaction on error
                raise e

    async def upload_and_process(self, files: List[UploadFile], project_id: int, bundle_entry: ImageBundleORM) -> List[ImageEntryORM]:
        """
        Uploads and processes a list of image files for a given project and image bundle.

        This method handles the creation of directories for original, transformed, and thumbnail images,
        saves the uploaded files, and performs image resizing operations.

        Args:
            files (List[UploadFile]): A list of image files to be uploaded.
            project_id (int): The unique identifier of the project.
            bundle_entry (ImageBundleORM): The image bundle ORM instance to which the images belong.

        Returns:
            List[ImageEntryORM]: A list of ORM instances representing the uploaded and processed images.
        """
        bundle_id = bundle_entry.id
        default_path = f"static/{project_id}_project/{bundle_id}_bundle/"

        # Full paths for each type of content
        upload_dir = os.path.join(self.base_dir, default_path, "original")
        transformed_dir = os.path.join(self.base_dir, default_path, "transformed")
        thumbnail_dir = os.path.join(self.base_dir, default_path, "thumbnail")

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(transformed_dir, exist_ok=True)
        os.makedirs(thumbnail_dir, exist_ok=True)

        image_entry_list = []
        for index, file in enumerate(files):
            file_extension = os.path.splitext(file.filename)[1].lower()
            file_name = f"{index}_{generate_random_filename(6)}{file_extension}"

            original_file = os.path.join(upload_dir, file_name)
            transformed_file = os.path.join(transformed_dir, file_name)
            thumbnail_file = os.path.join(thumbnail_dir, file_name)

            async with aiofiles.open(original_file, 'wb') as out_file:
                content = await file.read()  # Read async
                await out_file.write(content)  # Write async

            # Correct image orientation and resize (consider offloading to a thread for CPU-bound task)
            await asyncio.get_event_loop().run_in_executor(None, self.process_image, original_file, transformed_file, thumbnail_file)

            new_image = ImageEntryORM.create(
                bundle=bundle_entry,
                filename=file_name,
                path=default_path
            )
            image_entry_list.append(new_image)

        bundle_entry.status = BundleStatusEnum.UPLOADED
        bundle_entry.save()
        print(f"{colors.OKBLUE}Bundle #{bundle_entry.id} uploaded all {len(image_entry_list)} Images successfully.{colors.ENDC}")

        return image_entry_list  # Indicate success

    def list_bundles(self, _id: int) -> List[ImageBundleORM]:
        """
        Lists all image bundles associated with a given project ID.

        Args:
            _id (int): The unique identifier of the project.

        Returns:
            List[ImageBundleORM]: A list of image bundle ORM instances for the specified project.

        Raises:
            Exception: If the project with the given ID does not exist or any other db error occurs.
        """
        with self.db.atomic() as transaction:
            try:
                project = ProjectORM.get_by_id(_id)
                bundles = ImageBundleORM.select().where(ImageBundleORM.project == project).order_by(
                    ImageBundleORM.uploaded_at.desc())

                return bundles

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Project with ID {_id} does not exist.")
            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e

    def remove_bundle(self, bundle_id: int) -> None:
        """
        Removes an image bundle and all associated images and annotations from the database.

        Args:
            bundle_id (int): The unique identifier of the image bundle to be removed.

        Raises:
            Exception: If the bundle with the given ID does not exist.
        """
        with self.db.atomic() as transaction:
            try:
                bundle = ImageBundleORM.get_by_id(bundle_id)
                bundle_folder = None
                for image in bundle.images:
                    # Assuming Annotations are linked to Images, delete them first
                    bundle_folder = image.path

                    for annotation in image.annotations:
                        annotation.delete_instance()
                    image.delete_instance()

                if bundle_folder is not None:
                    remove_folder(bundle_folder)

                bundle.delete_instance()  # Delete the batch

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Bundle does not exist.")
            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e

    def process_image(self, original_file: str, transformed_file: str, thumbnail_file: str) -> None:
        """
        Corrects the image orientation and resizes it to the specified sizes.
        Args:
            original_file (str): The file path of the original image.
            transformed_file (str): The file path where the transformed image will be saved.
            thumbnail_file (str): The file path where the thumbnail image will be saved.
        """
        with Image.open(original_file) as img:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break

                exif = dict(img.getexif().items())

                if orientation in exif:
                    if exif[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img = img.rotate(90, expand=True)

                    # Remove orientation info from EXIF data
                    exif[orientation] = 1

                    # Convert EXIF data back to bytes
                    exif_bytes = piexif.dump({'0th': exif})

                    # Save the corrected image with updated EXIF data
                    img.save(original_file, exif=exif_bytes)
                else:
                    img.save(original_file)

            except (AttributeError, KeyError, IndexError) as e:
                # Cases: image doesn't have getexif, no EXIF orientation data, or invalid EXIF data
                print(f"{colors.FAIL}Failed to correct image orientation: {e}{colors.ENDC}")
                img.save(original_file)

            # Create transformed file in the desired size using resize_image method
            self.resize_image(img, transformed_file, Transformer.PROCESS_IMAGE_SIZE)

            # Create thumbnail
            self.resize_image(img, thumbnail_file, (64, 64))

    @staticmethod
    def resize_image(img: Image, output_path: str, size: Tuple[int, int] = (1024, 1024)) -> None:
        """
        Resizes an image to the specified size and saves it to the output path.

        Args:
            img (Image): The Image object to be resized.
            output_path (str): The file path where the resized image will be saved.
            size (tuple): The desired size of the image as a tuple (width, height).
        """
        img_copy = img.copy()
        img_copy.thumbnail(size)
        img_copy.save(output_path)

    def start_bundle_inference(self, image_list: List[ImageEntryORM], bundle_entry: ImageBundleORM, base_snapshot_id: int) -> None:
        """
        Starts the inference process on a bundle of images using a specified model snapshot.

        Args:
            image_list (list): A list of ImageEntryORM objects representing the images to be inferred.
            bundle_entry (ImageBundleORM): The image bundle ORM object associated with the images.
            base_snapshot_id (int): The unique identifier of the base snapshot to use for inference.

        Raises:
            Exception: If the base snapshot is not in a completed state or does not exist or if there is an error during the inference process or database transaction.

        """
        snapshot_path = None

        if base_snapshot_id is not None:
            try:
                # Load snapshot path from database
                snapshot = ModelSnapshotORM.get_by_id(base_snapshot_id)
                if snapshot.state is not SnapshotStatusEnum.COMPLETED.value:
                    raise Exception(f"The base snapshot with the id #{base_snapshot_id} is not in a completed state.")

                # Deselect all other snapshots for the same project
                ModelSnapshotORM.deselect_all(snapshot.project)
                # Select used snapshot as default snapshot
                snapshot.is_selected = True
                snapshot.save()

                # Load snapshot path from database
                snapshot_path = snapshot.path

            except DoesNotExist:
                raise Exception(f"The base snapshot with the id #{base_snapshot_id} does not exist.")
            except Exception as e:
                raise Exception(f"Error while starting image inference. Base snapshot error: {e}")

        start_time = self.current_milli_time()

        custom_model = CustomModel(base_snapshot_path=snapshot_path)
        self.ssd_model = custom_model.model.to(custom_model.device)  # Ensure model is on the correct device
        self.label_set = custom_model.label_set

        # Set the models to evaluation mode
        self.ssd_model.eval()

        project_id = bundle_entry.project.id
        bundle_id = bundle_entry.id

        with self.db.atomic() as transaction:

            try:
                # Get all images with batch_id
                annotation_count = 0
                image_array = []
                for image_entity in image_list:
                    transformed_image = get_image_path(image_entity, IMAGE_VERSION.TRANSFORMED, True)
                    tensor = self.get_image_tensor(transformed_image).to(custom_model.device)
                    results = self.inference(tensor)

                    # Add result to image
                    boxes_array = []
                    for detection in results:
                        x1, y1, x2, y2 = self.adjust_coordinates(detection.box)
                        boxes_array.append({
                            "box":(x1,y1,x2,y2),
                            "label":detection.label
                        })

                        AnnotationORM.create(
                            image=image_entity,
                            label=detection.label,
                            label_id=detection.label_id,
                            x=x1, y=y1,
                            width=x2 - x1, height=y2 - y1,
                            score=detection.score
                        )
                        annotation_count += 1

                    image_array.append({
                        "image_id": image_entity.id,
                        "image": f"static/{project_id}_project/{bundle_id}_bundle/transformed/{image_entity.filename}",
                        "annotations" : boxes_array
                    })

                duration = self.current_milli_time() - start_time
                # set Batch Status to Annotated
                bundle_entry.status = BundleStatusEnum.ANNOTATED
                bundle_entry.save()
                print(f"Found {colors.OKGREEN}{annotation_count} bounding boxes{colors.ENDC} on images")
                print(f"{colors.OKBLUE}Bundle #{bundle_id} annotated completely.{colors.ENDC} Status updated")

                self.save_inference_test(image_array, duration, project_id, bundle_id)

            except Exception as e:
                transaction.rollback()  # Roll back the transaction on error
                raise e

    @staticmethod
    def get_image_tensor(image_path: str) -> torch.Tensor:
        """
        Converts an image from a given path to a tensor suitable for model inference.

        Args:
            image_path (str): The file system path to the image.

        Returns:
            torch.Tensor: A tensor representation of the image.
        """
        image = Image.open(image_path).convert("RGB")
        t = Transformer()
        tensor, _ = t(image)  # Extract the tensor from the tuple
        return tensor.unsqueeze(0)

    def inference(self, image: torch.Tensor) -> Optional[List[InferenceResultDict]]:
        """
        Performs inference on a given image tensor using the loaded SSD model.

        Args:
            image (torch.Tensor): The image tensor to perform inference on.

        Returns:
            Optional[List[InferenceResultDict]]: A list of dictionaries containing the inference results, or None if the model is not loaded.
        """
        if self.ssd_model is None:
            return None
        # Turn off gradients to speed up this part
        with torch.no_grad():
            prediction = self.ssd_model(image)
            return self.read_tensor_result(prediction)

    def read_tensor_result(self, data: List[Dict]) -> List[InferenceResultDict]:
        """
        Reads the raw tensor results from the model inference and converts them into a list of annotations.

        Args:
            data (list): The raw output from the model inference.

        Returns:
            List[InferenceResultDict]: A list of dictionaries containing the processed inference results.
        """
        answer_list = []

        for i in range(len(data[0]['scores'])):
            if len(answer_list) >= self.max_results:
                print(f"{colors.WARNING}Ended annotation inference for image, because already {self.max_results} were{colors.ENDC}")
                break
            if data[0]['scores'][i] > self.min_confidence:  # Apply confidence threshold
                box = data[0]['boxes'][i]  # This is a tensor
                label_index = data[0]['labels'][i].item()
                label = self.label_set[label_index] if 0 <= label_index < len(self.label_set) else 'none'

                x1, y1, x2, y2 = self.adjust_coordinates(box.cpu().numpy() if box.is_cuda else box.numpy())

                answer_list.append(InferenceResultDict(
                    label=label,
                    label_id=label_index,
                    box=[x1, y1, x2, y2],
                    score=data[0]['scores'][i].item()
                ))

        return answer_list

    @staticmethod
    def current_milli_time():
        return round(time.time() * 1000)

    def save_inference_test(self, image_array, duration:int, project_id:int, bundle_id: int):
        json_data = {
            "bundle_id": bundle_id,
            "inference": {
                "duration" : duration,
                "result": image_array
            }
        }

        # Create the JSON file path
        json_file_path = os.path.join(self.base_dir, f"static/{project_id}_project/{bundle_id}_bundle/inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Write the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(
            f"{colors.OKGREEN}Test data for bundle #{bundle_id} has been saved successfully.{colors.ENDC}")

    @staticmethod
    def adjust_coordinates(box):
        target_size = Transformer.TRANSFORMED_IMAGE_SIZE
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), target_size[0]))
        y1 = max(0, min(int(y1), target_size[1]))
        x2 = max(0, min(int(x2), target_size[0]))
        y2 = max(0, min(int(y2), target_size[1]))
        return x1, y1, x2, y2

    def get_annotated_bundle(self, bundle_id: int) -> List[ImageEntryORM]:
        """
        Retrieves a list of image entries associated with a given bundle ID.

        Args:
            bundle_id (int): The unique identifier of the image bundle.

        Returns:
            List[ImageEntryORM]: A list of ImageEntryORM instances representing the images in the bundle.

        Raises:
            Exception: If the image bundle with the given ID does not exist.
        """
        with self.db.atomic() as transaction:
            try:
                # Try to Get Bundle
                image_bundle = ImageBundleORM.get_by_id(bundle_id)
                image_list = ImageEntryORM.select().where(ImageEntryORM.bundle == image_bundle)

                return image_list
            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Image bundle with ID {bundle_id} does not exist.")
            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e

    def update_annotation(self, image_id: int, annotation_array: List[AnnotationsReviewEntry]) -> List[AnnotationORM]:
        """
        Updates the annotations for a given image based on a provided array of annotations.

        Args:
            image_id (int): The unique identifier of the image to update.
            annotation_array (List[AnnotationsReviewEntry]): A list of annotations to apply to the image.

        Returns:
            bool: True if the update was successful, False otherwise.

        Raises:
            Exception: If the image with the given ID does not exist.
        """
        with self.db.atomic() as transaction:
            try:
                # Try to Get selected Image
                image = ImageEntryORM.get_by_id(image_id)

                # Existing annotation IDs for the image
                existing_annotation_ids = {annotation.id for annotation in image.annotations}

                new_array = []

                # Process each reviewed annotation
                for reviewed_annotation in annotation_array:
                    annotation_id = reviewed_annotation.id
                    
                    # If the annotation is already in the database, update it
                    if annotation_id in existing_annotation_ids:

                        annotation = AnnotationORM.get_by_id(annotation_id)
                        annotation.x = reviewed_annotation.box.x
                        annotation.y = reviewed_annotation.box.y
                        annotation.width = reviewed_annotation.box.width
                        annotation.height = reviewed_annotation.box.height
                        annotation.label = reviewed_annotation.label
                        annotation.active = reviewed_annotation.active
                        if annotation.original_state is True:
                            if reviewed_annotation.edited is True:
                                annotation.original_state = False
                        annotation.save()

                        new_array.append(annotation)

                    # If the annotation is new, create it
                    else:
                        newEntry = AnnotationORM.create(
                            image=image,
                            x=reviewed_annotation.box.x,
                            y=reviewed_annotation.box.y,
                            width=reviewed_annotation.box.width,
                            height=reviewed_annotation.box.height,
                            label=reviewed_annotation.label,
                            score=1,  # Assuming a default score of 1 for new annotations
                            from_inference=False,
                            original_state=False,
                            active=True
                        )
                        new_array.append(newEntry)

                # Check if Review state is done -> change bundle state
                self.check_complete_state(image)

                return new_array

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Image with ID {image_id} does not exist.")

            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e

    def check_complete_state(self, img:ImageEntryORM):
        with self.db.atomic() as transaction:
            try:
                # Set current image to corrected flag
                img.is_corrected = True
                img.save()

                # get Bundle of image
                img_bundle = img.bundle
                img_valid_counter = 0
                img_review_counter = 0
                for linked_img in img_bundle.images:
                    # Check if any annotation in linked_img.annotations has active=True
                    if any(annotation.active for annotation in linked_img.annotations):
                        # Increment valid image counter
                        img_valid_counter += 1

                        # If the image has been corrected, increment the review counter
                        if linked_img.is_corrected:
                            img_review_counter += 1

                print(f"There are {img_review_counter}/{img_valid_counter} Images reviewed")

                if img_review_counter >= img_valid_counter:
                    print(f"{colors.OKGREEN}Fully reviewed bundle{colors.ENDC}")
                    img_bundle.status = BundleStatusEnum.REVIEWED
                    img_bundle.save()

            except DoesNotExist:
                transaction.rollback()  # Roll back the transaction if the project does not exist
                raise Exception(f"Image with ID {img.id} does not exist.")

            except Exception as e:
                transaction.rollback()  # Roll back the transaction on other errors
                raise e

    def submit_test(self, bundle_id: int, duration: int, data: TestBundleInfoData) -> None:
        """
        Submits test data for a given bundle and saves it as a JSON file.

        Args:
            bundle_id (int): The unique identifier of the image bundle.
            duration (int): the ms duration seconds of the review session
            data (TestBundleInfoData): The test data to be submitted.

        Raises:
            Exception: If the bundle with the given ID does not exist or if there's an error writing the JSON file.
        """
        with self.db.atomic() as transaction:
            try:
                # Get the bundle
                bundle = ImageBundleORM.get_by_id(bundle_id)

                # Prepare the data structure
                json_data = {
                    "bundle_id": bundle_id,
                    "duration": duration,
                    "info": {
                        "boxCreatedCnt": data.box_created_cnt,
                        "boxRemovedCnt": data.box_removed_cnt,
                        "boxMovedCnt": data.box_moved_cnt,
                        "labelUpdatedCnt": data.label_updated_cnt,
                        "navigateImgCnt": data.navigated_img_cnt
                    },
                    "images": []
                }
                for image in bundle.images:
                    image_data = {
                        "image_id": image.id,
                        "image": f"static/{bundle.project.id}_project/{bundle_id}_bundle/transformed/{image.filename}",
                        "annotations": self.get_boxes_for_image(image),
                    }
                    json_data["images"].append(image_data)

                # Create the JSON file spath
                json_file_path = os.path.join(self.base_dir, f"static/{bundle.project.id}_project/{bundle_id}_bundle/_manual_labeling.json")

                # Write the JSON file
                with open(json_file_path, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)

                print(
                    f"{colors.OKGREEN}Test data for bundle #{bundle_id} has been saved successfully.{colors.ENDC}")

            except DoesNotExist:
                transaction.rollback()
                raise Exception(f"Bundle with ID {bundle_id} does not exist.")
            except Exception as e:
                transaction.rollback()
                raise Exception(f"Error submitting test data: {str(e)}")

    def get_boxes_for_image(self, image: ImageEntryORM):
        """Helper method to get all bounding boxes for an image."""
        return [
            {
            'box': (int(ann.x), int(ann.y), int(ann.x + ann.width), int(ann.y + ann.height)),
            'label': ann.label
            }
            for ann in image.annotations
            if ann.active
        ]
