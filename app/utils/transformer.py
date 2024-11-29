import os
import random
import torch
from torchvision import transforms
from dotenv import load_dotenv
from app.models.dicts.training import LabelSetDict
from app.utils.augmentation import RandomZoomInOut, RandomHorizontalFlip, RandomVerticalFlip
from typing import Tuple, List
from PIL import Image, ImageDraw

from app.utils.image_padding import ImageSquare, ImageScale
from app.utils import colors


class Transformer:
    """
    A transformer class for image preprocessing which includes resizing, padding, 
    converting to tensor, and normalizing the image.

    Attributes:
        MEAN (List): The mean values for normalization of images.
        STD (List): The standard deviation values for normalization of images.
        TRANSFORMED_IMAGE_SIZE (tuple): The desired image size after resizing and padding.
    """

    # Define class variables
    MEAN: List[float] = [0.485, 0.456, 0.406]
    STD: List[float] = [0.229, 0.224, 0.225]
    TRANSFORMED_IMAGE_SIZE: Tuple[int, int] = (300, 300)
    PROCESS_IMAGE_SIZE: Tuple[int, int] = (640, 640)

    def __init__(self, train: bool = False) -> None:
        """
        Initializes the Transformer with a ResizeAndPad instance, a ToTensor instance,
        and a Normalize instance using the class variables for image size, mean, and std.
        """
        self.pad_square = ImageSquare(self.TRANSFORMED_IMAGE_SIZE)
        self.scale_to_target = ImageScale(self.TRANSFORMED_IMAGE_SIZE)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.MEAN, std=self.STD)
        self.random_zoom_in_out = RandomZoomInOut()
        self.random_horizontal_flip = RandomHorizontalFlip()
        self.random_vertical_flip = RandomVerticalFlip()
        self.train = train

        load_dotenv()
        base_dir = os.getenv('BASE_DIR')
        self.save_dir = os.path.join(base_dir, 'static/dump/')

    @staticmethod
    def visual_transforms(image):
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
        image = color_jitter(image)

        random_grayscale = transforms.RandomGrayscale(p=0.2)
        image = random_grayscale(image)

        # New color transformations
        if random.random() < 0.3:
            # Randomly invert colors
            image = transforms.functional.invert(image)

        if random.random() < 0.3:
            # Randomly adjust hue more aggressively
            hue_factor = random.uniform(-0.5, 0.5)
            image = transforms.functional.adjust_hue(image, hue_factor)

        # Existing sharpness and blur adjustments
        image = transforms.functional.adjust_sharpness(image, sharpness_factor=random.uniform(0.5, 2.0))
        image = transforms.functional.gaussian_blur(image, kernel_size=random.choice([3, 5]), sigma=[0.1, 1.0])

        return image

    def save_image_with_boxes(self, image: torch.Tensor, target: LabelSetDict, filename: str):
        """
        Saves the image with bounding boxes drawn on it.

        Args:
            image (torch.Tensor): The image tensor.
            target (LabelSetDict): The target containing bounding boxes.
            filename (str): The filename to save the image as.
        """
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(image)
        draw = ImageDraw.Draw(img)

        # Draw bounding boxes
        for box in target.boxes:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

        # Create a unique filename if the file already exists
        save_path = os.path.join(self.save_dir, filename)
        base, ext = os.path.splitext(save_path)
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base}_{counter}{ext}"
            counter += 1

        # Save the image
        img.save(save_path)
        #print(f"Saved augmented image with bounding boxes: {save_path}")


    def __call__(self, image: Image, target: LabelSetDict = None):
        """
        Applies the transformations to the image and target.

        Args:
            image (PIL.Image): The image to be transformed.
            target (dict, optional): The target to be transformed alongside the image.

        Returns:
            tuple: A tuple containing the transformed image and target.
        """
        # Pad PIL Image to square (disregarding resolution)
        # Scale All bounding Boxes coordinates to current image size (bounding box always max 300x300)
        image, target = self.pad_square(image, target)
        image = self.to_tensor(image)

        if image.shape[-2] == 0 or image.shape[-1] == 0:
            print(f"{colors.WARNINGN}Warning: Encountered image with zero width or height after padding. Skipping transformations.{colors.ENDC}")
            return image, target

        if self.train:
            image, target = self.random_zoom_in_out(image, target)
            image, target = self.random_horizontal_flip(image, target)
            image, target = self.random_vertical_flip(image, target)
            image = self.visual_transforms(image)

        image, target = self.scale_to_target(image, target)

        if target is not None and target.boxes is not None and target.labels is not None:
            valid_indices = [idx for idx, bx in enumerate(target.boxes) if self.is_valid_box(bx)]

            if valid_indices:
                target = LabelSetDict(
                    boxes=target.boxes[valid_indices],
                    labels=target.labels[valid_indices]
                )
            elif self.train:
                # keine Bounding Boxen mehr vorhanden -> fallback return
                print(f"{colors.WARNING}using trainings data without augmentation{colors.ENDC}")
                return self.scale_to_target(self.normalize(image), target)

        image = self.normalize(image)

        #if self.train:
           #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           #self.save_image_with_boxes(image, target, f"{timestamp}.jpg")

        return image, target

    @staticmethod
    def is_valid_box(box):
        x1, y1, x2, y2 = box

        if x2 < x1:
            print(f"{colors.FAIL}Box vertical mirror{colors.ENDC}")
            return False
        if y2 < y1:
            print(f"{colors.FAIL}Box horizontal mirror{colors.ENDC}")
            return False
        if (x1 < 0 or x2 > 300) or (y1 < 0 or y2 > 300):
            print(f"{colors.FAIL}Box invalid bounds{colors.ENDC}")
            return False

        if x2 - x1 < 2:
            print(f"{colors.FAIL}Box width to small{colors.ENDC}")
            return False
        if y2 - y1 < 2:
            print(f"{colors.FAIL}Box height to small{colors.ENDC}")
            return False

        return True

    @staticmethod
    def deep_copy_target(target):
        if target is None:
            return None
        return LabelSetDict(
            boxes=target.boxes.clone() if hasattr(target, 'boxes') else torch.empty(0, 4),
            labels=target.labels.clone() if hasattr(target, 'labels') else torch.empty(0, dtype=torch.long)
        )

