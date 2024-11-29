from typing import List, Tuple

from torch.utils.data import Dataset
from PIL import Image
from torch import Tensor

from app.models.dicts.training import LabelSetDict

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'hair brush'
]

class CustomDataset(Dataset):
    """
    A custom dataset class that inherits from PyTorch's Dataset class. This class is designed to handle
    datasets for object detection tasks, where each item in the dataset consists of an image and its
    corresponding annotations (e.g., bounding boxes and labels).

    Attributes:
        image_paths (list): A list of file paths to the images.
        annotations (list): A list of annotations corresponding to each image.
        transform (callable, optional): An optional transform to be applied on a sample.
    """

    def __init__(self, image_paths: List[str], annotations: List[LabelSetDict], transformer=None) -> None:
        """
        Initializes the CustomDataset instance with image paths, annotations, and an optional transformer.

        Args:
            image_paths (list): A list of file paths to the images.
            annotations (list): A list of annotations corresponding to each image.
            transformer (callable, optional): An optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transformer

    def __getitem__(self, idx: int) -> Tuple[Tensor, LabelSetDict]:
        """
        Retrieves the image and its annotations at the specified index in the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its annotations after any transformations have been applied.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        target = self.annotations[idx]

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.image_paths)

