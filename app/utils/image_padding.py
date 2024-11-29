from typing import Tuple
from PIL import Image
import torch

from app.models.dicts.training import LabelSetDict


class ImageSquare:
    """
    A class that pads an image to a square shape and scales the bounding box coordinates.

    Attributes:
        output_size (int): The size of the output square image.

    Methods:
        __call__(image, target=None): Pads the input image to a square shape and scales the bounding box coordinates.
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        """
        Initializes the ResizeAndPad transformer with a specified target size.

        Args:
            size (tuple of int): The target size (height, width) as a tuple.
        """
        self.size = size[0]

    def __call__(self, image: Image.Image, target: LabelSetDict = None) -> Tuple[Image.Image, LabelSetDict]:
        """
        Pads the input image to a square shape and scales the bounding box coordinates.

        Args:
            image (PIL.Image): The image to be padded and resized.
            target (dict, optional): A dictionary containing target data with bounding boxes under the key 'boxes'.
                                     The bounding boxes are expected to be a tensor of shape (N, 4), where N is the
                                     number of bounding boxes, and the 4 elements are the coordinates (x1, y1, x2, y2).
                                     Default is None, which means no target transformation is done.

        Returns:
            tuple: A tuple containing the padded image and the (optionally) transformed target dictionary.
        """
        w, h = image.size
        # print(f"\n\n---\nOriginal image size: {w} x {h}")

        # Calculate padding dimensions
        max_dim = max(w, h)
        pad_x, pad_y = (max_dim - w) // 2, (max_dim - h) // 2

        # Pad the image
        padded_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        padded_image.paste(image, (pad_x, pad_y))

        if target is not None and 'boxes' in target:
            target = target.copy()
            boxes = target.boxes
            #print(f"Original bounding boxes for 300x300 image:\n{boxes}")

            # Scale bounding box coordinates to match the padded image size relative to the reference size
            scale_factor = max_dim / self.size

            boxes[:, :4] *= scale_factor
            # Clamp the bounding box coordinates to be within the valid range
            boxes[:, :4] = torch.clamp(boxes[:, :4], min=0, max=max_dim)

            #print(f"Scale up (x{scale_factor}) bounding boxes for {max_dim}x{max_dim} image:\n{boxes}")

            target.boxes = boxes

        return padded_image, target


class ImageScale:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size[0]

    def __call__(self, image: torch.Tensor, target: LabelSetDict = None) -> Tuple[torch.Tensor, LabelSetDict]:
        """
        Scales the input image tensor and the bounding box coordinates to a specified size.

        Args:
            image (torch.Tensor): The input image tensor to be scaled.
            target (LabelSetDict, optional): A dictionary containing target data with bounding boxes under the key 'boxes'.
                                     The bounding boxes are expected to be a tensor of shape (N, 4), where N is the
                                     number of bounding boxes, and the 4 elements are the coordinates (x1, y1, x2, y2).
                                     Default is None, which means no target transformation is done.
        """
        current_size = image.shape[-1]

        # Calculate the scale factor
        scale_factor = self.size / current_size
        # print(f"Final scale down ({current_size} -> {self.size})\nfactor: {scale_factor}")

        # Scale the image tensor
        scaled_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        if target is not None and 'boxes' in target:
            target = target.copy()
            boxes = target.boxes
            # Scale the bounding box coordinates
            boxes[:, :4] *= scale_factor

            # Handle cases where x2 < x1 or y2 < y1
            boxes[:, 0], boxes[:, 2] = torch.min(boxes[:, 0], boxes[:, 2]), torch.max(boxes[:, 0], boxes[:, 2])
            boxes[:, 1], boxes[:, 3] = torch.min(boxes[:, 1], boxes[:, 3]), torch.max(boxes[:, 1], boxes[:, 3])

            # Adjust the bounding box coordinates to maintain a minimum size
            boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 2)
            boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 2)

            # Clamp the bounding box coordinates to be within the valid range
            boxes[:, :4] = torch.clamp(boxes[:, :4], min=0, max=self.size)

            target.boxes = boxes

        return scaled_image, target
