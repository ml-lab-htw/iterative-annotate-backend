import json
import os
from math import isnan
from typing import List, Tuple, Optional
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dotenv import load_dotenv
from playhouse.pool import PooledMySQLDatabase
from torch.utils.data import DataLoader
from peewee import DoesNotExist
from app.utils import colors

from app.models.database import ImageEntryORM, ImageBundleORM, AnnotationORM, ModelSnapshotORM, SnapshotBundleLink, ProjectORM
from app.models.dicts.training import LabelSetDict
from app.utils.custom_dataset import CustomDataset, COCO_INSTANCE_CATEGORY_NAMES
from app.utils.custom_model import CustomModel
from app.utils.filename import generate_random_filename
from app.enums.snapshot_enum import SnapshotStatusEnum
from app.utils.image_files import get_image_path, IMAGE_VERSION
from app.utils.transformer import Transformer


class TrainingsService:
    """
    Service class for managing training sessions and snapshots for machine learning models.

    Attributes:
        db (PooledMySQLDatabase): The database connection pool.
        learning_rate (float): The learning rate for the training session.
        batch_size (int): The batch size for the training session.
        epochs (int): The number of epochs for the training session.
        stopped (bool): Flag to indicate if the training has been stopped.
        snapshot_orm (ModelSnapshotORM): The ORM object for the current snapshot.
        snapshot_nbr (int): The number of the current snapshot.
        custom_model (CustomModel): The custom model object for training.
        model (torch.nn.Module): The PyTorch model for training.
        dataloader (DataLoader): The DataLoader for the training dataset.
        project_orm (ProjectORM): The ORM object for the project associated with the training.
    """

    def __init__(self, db: PooledMySQLDatabase, learning_rate: float, batch_size: int, epochs: int) -> None:
        """
        Initializes the TrainingsService with the given database connection and training parameters.

        Args:
            db (PooledMySQLDatabase): The database connection pool.
            learning_rate (float): The learning rate for the training session.
            batch_size (int): The batch size for the training session.
            epochs (int): The number of epochs for the training session.
        """
        self.db = db
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.stopped = False

        load_dotenv()
        self.base_dir = os.getenv('BASE_DIR')

        self.snapshot_orm = None
        self.snapshot_nbr = 1
        self.custom_model = None
        self.model = None
        self.dataloader = None
        self.project_orm = None

    def start_snapshot(self, snapshot_name: str, base_snapshot_id: int, bundle_id_array: List[int]) -> ModelSnapshotORM:
        if self.snapshot_orm is not None:
            raise Exception("Snapshot creation already initialized. Aborting...")

        try:
            # Step 1: Determine the initial label set from json file or coco base
            label_set, model_path = self.load_label_set(base_snapshot_id)
            print(model_path)
            print(f"Label set before {len(label_set)}:")
            #print(label_set)

            # Step 2: Fetch the training data from db and update label set with any new labels from review circle
            included_bundles, project = self.get_bundle_array(bundle_id_array)
            updated_label_set = self.update_label_set_with_new_labels(label_set, included_bundles)
            print(f"Label set after {len(updated_label_set)}:")
            print(updated_label_set)

            output_nodes = len(updated_label_set)

            # Step 3: Initialize the model with the complete and updated label set
            self.custom_model = CustomModel(base_snapshot_path=model_path, num_classes=output_nodes)
            self.custom_model.label_set = list(updated_label_set)  # Ensure the model's label set is updated
            self.model = self.custom_model.model

            self.project_orm = project

            self.dataloader = self.load_training_dataset(included_bundles)

            # Step 4: Create database entry for the new snapshot
            if snapshot_name is None:
                snapshot_name = f"Snapshot {self.snapshot_nbr}"

            self.snapshot_orm = ModelSnapshotORM.create(
                project=self.project_orm,
                name=snapshot_name,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )

            # Link all bundles to the new snapshot
            for bundle in included_bundles:
                SnapshotBundleLink.create(
                    snapshot=self.snapshot_orm,
                    bundle=bundle
                )

            return self.snapshot_orm

        except Exception as e:
            raise Exception(f"Error while preparing data for training: {e}")

    def load_label_set(self, base_snapshot_id: Optional[int]) -> (List[str], str):
        if base_snapshot_id:
            snapshot = ModelSnapshotORM.get_by_id(base_snapshot_id)
            if snapshot and snapshot.state == SnapshotStatusEnum.COMPLETED.value:
                model_file = os.path.join(self.base_dir, snapshot.path)
                label_set_file = model_file.replace('.pth', '_labels.json')
                if os.path.exists(label_set_file) and os.path.exists(model_file):
                    with open(label_set_file, 'r') as file:
                        return json.load(file), snapshot.path

        return COCO_INSTANCE_CATEGORY_NAMES.copy(), None  # Return a copy of the default label set

    def update_label_set_with_new_labels(self, label_list: List[str], bundles: List[ImageBundleORM]) -> List[str]:
        current_labels = set(label.lower() for label in label_list)  # Use a set temporarily for fast lookup
        for bundle in bundles:
            for image in bundle.images:
                for annotation in image.annotations:
                    if annotation.active:
                        lower_label = annotation.label.lower()
                        if lower_label not in current_labels:
                            label_list.append(lower_label)
                            current_labels.add(lower_label)
        return label_list

    @staticmethod
    def get_bundle_array(bundle_ids: List[int]) -> Tuple[List[ImageBundleORM], Optional[ProjectORM]]:
        """
        Retrieves a list of ImageBundleORM objects and the associated ProjectORM object for the given bundle IDs.

        Args:
            bundle_ids (List[int]): A list of bundle IDs to retrieve.

        Returns:
            Tuple[List[ImageBundleORM], Optional[ProjectORM]]: A tuple containing a list of ImageBundleORM objects
            and an optional ProjectORM object if found.

        Raises:
            Exception: If one of the specified image batches does not exist or if no project is specified for the image batch.
        """
        bundle_orm_list = []
        project_orm = None
        try:
            for bundle_id in bundle_ids:
                bundle_select = ImageBundleORM.select().where(ImageBundleORM.id == bundle_id)
                bundle_orm_list.extend(bundle_select)

                if project_orm is None and bundle_select.count() > 0:
                    project_orm = bundle_select[0].project

        except DoesNotExist:
            raise Exception("One of the specified image batches does not exist.")
        except Exception as e:
            raise Exception(f"Error while preparing data for training: {e}")

        if project_orm is None:
            raise Exception("No project specified for the specified image batch")

        return bundle_orm_list, project_orm

    def load_training_dataset(self, bundle_array: List[ImageBundleORM]) -> DataLoader:
        """
        Loads the training dataset from a list of image bundles.

        Args:
            bundle_array (List[ImageBundleORM]): A list of ImageBundleORM objects to load the dataset from.

        Returns:
            DataLoader: A DataLoader object containing the dataset for training.

        Raises:
            Exception: If an image bundle does not exist or there is an error while preparing the data for training.
        """
        try:
            image_paths = []
            labels = []

            for bundle in bundle_array:
                image_entries = ImageEntryORM.select().where(ImageEntryORM.bundle == bundle)

                for img in image_entries:
                    annotations = AnnotationORM.select().where(AnnotationORM.image == img).where(AnnotationORM.active == True)

                    if annotations.count() < 1:
                        print("No annotations for image -> Continue")
                        continue
                    else:
                        print(f"{annotations.count()} annotations on image -> add to training")

                    formatted_labels = self.orm_to_labelset(annotations)

                    path_to_file = get_image_path(img, IMAGE_VERSION.TRANSFORMED, True)
                    if not os.path.exists(path_to_file):
                        print("Images are not available")
                        continue

                    image_paths.append(path_to_file)
                    labels.append(formatted_labels)

            # Create a single transform instance
            dataset = CustomDataset(image_paths, labels, transformer=Transformer(train=True))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

        except DoesNotExist:
            raise Exception(f"Image bundle does not exist.")
        except Exception as e:
            raise Exception(f"Error while preparing data for training: {e}")
        
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, LabelSetDict]]) -> Tuple[torch.Tensor, List[LabelSetDict]]:
        """
        Collates a batch of data into a format suitable for training.

        Args:
            batch: A list of tuples, each containing an image tensor and its corresponding label set dictionary.

        Returns:
            Tuple[torch.Tensor, List[LabelSetDict]]: A tuple containing a batch of image tensors and a list of label set dictionaries.
        """
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(targets)

    @staticmethod
    def current_milli_time():
        return round(time.time() * 1000)

    def train_model(self, use_auto_lr = False, use_adam_optimizer = False) -> None:
        """
        Trains the model using the loaded dataset.

        Raises:
            Exception: If the snapshot ORM object is not created or if there is an error during the training process.
        """

        if self.snapshot_orm is None:
            raise Exception(f"Something went wrong in the snapshot creation process. Aborted")

        start_ms = self.current_milli_time()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) if use_adam_optimizer else optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.35) if use_auto_lr else None # Adjust step_size and gamma as needed
        self.model.train()

        # Set status to running
        self.change_snapshot_status(SnapshotStatusEnum.STARTED)

        loss_array = []

        for epoch in range(self.epochs):
            loss_average = []
            for images, targets in self.dataloader:
                try:
                    images = images.to(self.custom_model.device)
                    targets = [{
                        'boxes': t.boxes.to(self.custom_model.device),
                        'labels': t.labels.to(self.custom_model.device)
                    } for t in targets]

                    optimizer.zero_grad()
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    if isnan(losses.item()):
                        continue

                    loss_average.append(losses.item())
                    losses.backward()
                    optimizer.step()

                except Exception as e:
                    self.save_loss_to_db(loss_array)
                    self.change_snapshot_status(SnapshotStatusEnum.ABORTED)
                    raise Exception(f"Training failed: {e}")

            # Scheduler updates the learning rate
            if use_auto_lr and scheduler is not None:
                scheduler.step()

            if len(loss_average) < 1:
                print(f"{colors.FAIL}Epoch failed because loss function is out of number bounds{colors.ENDC}")
                continue

            loss_avg = sum(loss_average) / len(loss_average)
            loss_string = f"{loss_avg:.4f}"
            loss_array.append(loss_string)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{self.epochs}] | Loss {colors.OKCYAN}{loss_avg}{colors.ENDC} | lr: {current_lr}")

            if (epoch+1) % 5 == 0:
                self.save_loss_to_db(loss_array)

            if self.stopped:
                break

        end_ms =  self.current_milli_time()
        dur_ms = end_ms - start_ms

        print(f"{colors.OKGREEN}The model training took {dur_ms} ms to train{colors.ENDC}")

        # Save loss Array to db
        self.save_loss_to_db(loss_array)

        # Save snapshot file
        self.save_model_weights()

        # Save Trainings REVIEW DATA
        json_data = {
            "trainings_info": {
                    "duration": dur_ms,
                    "learningRate": self.learning_rate,
                    "finalLoss" : loss_avg,
                    "normalizedLoss" : 0.0,
                    "totalChange" : -0.0,
                    "batchSize": self.batch_size,
                    "epochs" : self.epochs,
                    "optimizer" : "SGD",
                    "images": 10
            },
        }
        self.save_trainings_info(json_data)

    def save_trainings_info(self, data):
        json_file_path = os.path.join(self.base_dir, f"static/{self.project_orm.id}_project/snapshots/{self.snapshot_orm.id}_train_results.json")

        # Write the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


    def orm_to_labelset(self, label_orms: List[AnnotationORM]) -> LabelSetDict:
        """
        Converts ORM objects to a label set dictionary suitable for training.

        Args:
            label_orms (List[AnnotationORM]): A list of AnnotationORM objects to convert.

        Returns:
            LabelSetDict: A dictionary containing the converted label set with bounding boxes and label IDs.
        """
        boxes = []
        label_ids = []
        for annotation in label_orms:

            if not annotation.active:
                continue

            x_min = int(annotation.x)
            y_min = int(annotation.y)
            x_max = (int(annotation.x) + int(annotation.width))
            y_max = (int(annotation.y) + int(annotation.height))

            # Use the existing label set to find the correct label index
            try:
                label_index = self.custom_model.label_set.index(annotation.label)

                label_ids.append(label_index)
                boxes.append([x_min, y_min, x_max, y_max])

            except ValueError:
                print(f"Label '{annotation.label}' not found in the model's label set. Check label synchronization.")

        return LabelSetDict(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(label_ids, dtype=torch.int64)
        )

    def save_loss_to_db(self, loss_array):
        with self.db.atomic() as transaction:
            try:
                self.snapshot_orm.loss = "|".join(loss_array)
                self.snapshot_orm.save()
            except Exception as e:
                transaction.rollback()

    def save_model_weights(self):
        """
        Saves the model weights and label set to files.

        Raises:
            Exception: If there is an error while updating the snapshot status in the database.
        """
        # Save the model state
        default_path = f"static/{self.project_orm.id}_project/snapshots/"
        snapshot_dir = os.path.join(self.base_dir, default_path)

        os.makedirs(snapshot_dir, exist_ok=True)

        file_name = f"{self.snapshot_orm.id}_{generate_random_filename(6)}.pth"
        file_path = os.path.join(snapshot_dir, file_name)
        db_path = os.path.join(default_path, file_name)

        # Save only the model weights
        torch.save(self.model.state_dict(), file_path)
        print(f"Model weights saved to file: {file_path}")

        # Save the label set
        label_set_file = file_path.replace('.pth', '_labels.json')
        with open(label_set_file, 'w') as f:
            json.dump(self.custom_model.label_set, f)
            print(f"Label file saved: {label_set_file}")

        self.change_snapshot_status(SnapshotStatusEnum.COMPLETED, db_path)
        print(f"Database updated successfully")

    def change_snapshot_status(self, new_status: SnapshotStatusEnum, add_path: Optional[str] = None) -> None:
        """
        Changes the snapshot status in the database.

        Args:
            new_status (SnapshotStatusEnum): The new status to set for the snapshot.
            add_path (str, optional): The path to add to the snapshot ORM object.

        Raises:
            Exception: If there is an error while updating the snapshot status in the database.
        """
        with self.db.atomic() as transaction:
            try:
                if add_path:
                    self.snapshot_orm.path = add_path
                self.snapshot_orm.state = new_status
                self.snapshot_orm.save()
            except Exception as e:
                transaction.rollback()
                raise Exception(f"Error while creating new snapshot entry: {e}")

    def abort_snapshot(self) -> None:
        """
        Aborts the snapshot process and sets the snapshot status to aborted.
        """
        if self.snapshot_orm is None:
            return

        # Set training stopping flag
        self.stopped = True
        self.change_snapshot_status(SnapshotStatusEnum.ABORTED)