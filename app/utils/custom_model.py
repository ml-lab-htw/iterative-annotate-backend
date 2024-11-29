import json
import os
from dotenv import load_dotenv
import torch
import torchvision.models.detection as models
from app.utils.custom_dataset import COCO_INSTANCE_CATEGORY_NAMES
from app.utils.transformer import Transformer

from app.utils import colors


class CustomModel:
    """
    A custom model class that encapsulates the model architecture, loading, and adjustment of a pre-trained object detection model.

    Attributes:
        img_size (int): The size of the images that the model expects.
        device (torch.device): The device (CPU or GPU) on which the model will be loaded and run.
        label_set (list): A list of category names for object detection.
        num_classes (int): The number of classes for object detection.
        model (torch.nn.Module): The loaded PyTorch model.

    Args:
        base_snapshot_path (str): The file path to the base snapshot of the model weights.
    """

    def __init__(self, base_snapshot_path: str = None, num_classes: int = None) -> None:
        """
        Initializes the CustomModel instance with a specified snapshot path for the base model weights.

        This method sets up the image size, device, label set, and number of classes for the model. It also creates the model
        by loading the weights from the given snapshot path and adjusts the model's image transformation settings.

        Args:
            base_snapshot_path (str): The file path to the base snapshot of the model weights.
            num_classes (int): Overwrites the output nodes for training
        """
        load_dotenv()
        self.base_dir = os.getenv('BASE_DIR')

        self.img_size = Transformer.TRANSFORMED_IMAGE_SIZE
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.inference = num_classes is None

        if self.inference:
            print(f"{colors.OKGREEN}Starting model for inference{colors.ENDC}")

            if base_snapshot_path is not None:
                # with base model files
                print(f"{colors.OKBLUE}Loading custom model files{colors.ENDC}")
                snapshot_dir = os.path.join(self.base_dir, base_snapshot_path)
                self.label_set = self.load_label_file(snapshot_dir)

                output_classes = len(self.label_set)

                print(f"...initializing with {colors.OKBLUE}{output_classes} output nodes{colors.ENDC}")

                self.model = models.ssd300_vgg16(num_classes=output_classes).to(self.device)
                self.load_model_weights(base_snapshot_path)
            else:
                # with coco base
                print(f"{colors.OKBLUE}Using default COCO weights and labels{colors.ENDC}")
                self.label_set = COCO_INSTANCE_CATEGORY_NAMES.copy()
                self.model = models.ssd300_vgg16(weights=models.SSD300_VGG16_Weights.COCO_V1).to(self.device)

        else:
            print(f"{colors.OKGREEN}Starting model for training{colors.ENDC}")
            self.label_set = None  # This will need to be set based on the training setup
            self.model = models.ssd300_vgg16(num_classes=num_classes).to(self.device)

            if base_snapshot_path is not None:
                # mit Basis
                print(f"{colors.OKBLUE}Loading custom model files and combining {num_classes} labels{colors.ENDC}")
                self.load_model_weights(base_snapshot_path)
            else:
                # ohne Basis
                print(f"{colors.OKBLUE}Using default COCO weights but customizing for {num_classes} labels{colors.ENDC}")

        # Adjust the model's image transformation settings if needed
        self.model.transform.image_mean = Transformer.MEAN
        self.model.transform.image_std = Transformer.STD
        self.model = self.model.to(self.device)
        print(f"Initialized SSD VGG16 on {colors.OKGREEN}{self.device}{colors.ENDC}")

    def load_model_weights(self, path):
        if path is not None:
            snapshot_dir = os.path.join(self.base_dir, path)
            state_dict = torch.load(snapshot_dir, map_location=self.device)

            # Filter out the incompatible weights from the loaded state_dict
            compatible_state_dict = {k: v for k, v in state_dict.items() if
                                     k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}

            # Load the filtered compatible weights
            self.model.load_state_dict(compatible_state_dict, strict=False)

            # Re-initialize the layers for which the weights were not loaded (typically the classification head)
            for k, v in self.model.state_dict().items():
                if k not in compatible_state_dict:
                    print(f"{colors.WARNING}Re-initializing layer{colors.ENDC}: {k}")
                    if 'weight' in k:
                        torch.nn.init.xavier_uniform_(v)
                    elif 'bias' in k:
                        torch.nn.init.constant_(v, 0)


    @staticmethod
    def load_label_file(full_path: str):
        label_set_file = full_path.replace('.pth', '_labels.json')

        if os.path.exists(label_set_file):
            with open(label_set_file, 'r') as f:
                return json.load(f)

