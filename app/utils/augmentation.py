import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F
from typing import Tuple, Optional
from app.models.dicts.training import LabelSetDict
from app.utils import colors


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    def forward(self, image: Tensor, target: LabelSetDict = None) -> Tuple[Tensor, LabelSetDict]:
        if torch.rand(1).item() < self.p:
            image = F.hflip(image)
            if target is not None and target.boxes is not None:
                width = image.shape[-1]
                target = target.copy()
                target.boxes[:, [0, 2]] = width - target.boxes[:, [2, 0]]
                target.boxes = target.boxes.clamp(0, width)
        return image, target


class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, image: Tensor, target: LabelSetDict = None) -> Tuple[Tensor, LabelSetDict]:
        if torch.rand(1).item() < self.p:
            image = F.vflip(image)
            if target is not None and target.boxes is not None:
                height = image.shape[-2]
                target = target.copy()
                target.boxes[:, [1, 3]] = height - target.boxes[:, [3, 1]]
                target.boxes = target.boxes.clamp(0, height)
        return image, target


class RandomZoomInOut(nn.Module):
    def __init__(self, p: float = 0.5, zoom_out_prob: float = 0.5, size_range: Tuple[float, float] = (0.5, 1.0)):
        super().__init__()
        self.p = p
        self.zoom_out_prob = zoom_out_prob
        self.size_range = size_range
        self.image_size = 300

    def forward(self, image: Tensor, target: Optional[LabelSetDict] = None) -> Tuple[Tensor, Optional[LabelSetDict]]:
        if torch.rand(1).item() > self.p:
            return image, target

        if torch.rand(1).item() < self.zoom_out_prob:
            return self.zoom_out(image, target)
        else:
            return self.zoom_in(image, target)

    def zoom_out(self, image: torch.Tensor, target: Optional[LabelSetDict] = None) -> Tuple[torch.Tensor, Optional[LabelSetDict]]:
        num_channels = image.shape[0]
        original_size = image.shape[1]  # Assuming the input image is square (width = height)
        new_image = torch.randint(0, 256, (num_channels, original_size, original_size), dtype=torch.float32) / 255.0

        size_percentage = torch.rand(1).item() * (self.size_range[1] - self.size_range[0]) + self.size_range[0]
        new_side_length = int(original_size * size_percentage)

        max_x = original_size - new_side_length
        max_y = original_size - new_side_length

        left = torch.randint(0, max_x + 1, (1,)).item()
        top = torch.randint(0, max_y + 1, (1,)).item()

        right = left + new_side_length
        bottom = top + new_side_length

        new_image[:, top:bottom, left:right] = F.resize(image, (new_side_length, new_side_length))

        if target is not None and 'boxes' in target:
            scale_factor = new_side_length / original_size
            target = target.copy()
            boxes = target.boxes * scale_factor
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            boxes = boxes.clamp(0, original_size - 1)
            boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1)
            boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1)
            target.boxes = boxes
            return new_image, target

        return new_image, target

    def zoom_in(self, image: torch.Tensor, target: Optional[LabelSetDict] = None) -> Tuple[torch.Tensor, Optional[LabelSetDict]]:
        # 1. Get side length of image tensor
        original_side_length = image.shape[-1]

        # 2. Select a random box from the target.boxes array
        box_index = torch.randint(0, len(target.boxes), (1,)).item()
        selected_box = target.boxes[box_index]

        # 3. Set the Max zoom factor
        box_width = selected_box[2] - selected_box[0]
        box_height = selected_box[3] - selected_box[1]
        bigger_side = max(box_width, box_height)
        min_crop_size = int(bigger_side * 1.05) if original_side_length / bigger_side > 1.05 else int(bigger_side + 5) # Ensure the box fits comfortably

        # Handle case where min_crop_size is larger than the original image
        if min_crop_size >= original_side_length:
            print(f"{colors.WARNING}Cannot zoom in further: box is too large relative to image size{colors.ENDC}")
            print(f"original: image: {original_side_length} < box:{min_crop_size}")
            print(f"bound box: {box_width}, {box_height}")
            return image, target  # Return original image and target without modification

        max_crop_size = original_side_length

        # Choose a random crop size between min_crop_size and max_crop_size
        new_side_length = torch.randint(min_crop_size, max_crop_size + 1, (1,)).item()

        # Calculate the center of the selected box
        box_center_x = (selected_box[0] + selected_box[2]) / 2
        box_center_y = (selected_box[1] + selected_box[3]) / 2

        # Calculate crop boundaries
        left = max(0, int(box_center_x - new_side_length / 2))
        top = max(0, int(box_center_y - new_side_length / 2))

        # Adjust left and top to ensure we don't go out of bounds
        left = min(left, original_side_length - new_side_length)
        top = min(top, original_side_length - new_side_length)

        cropped_image = F.crop(image, top, left, new_side_length, new_side_length)

        if target is not None and 'boxes' in target and 'labels' in target:
            scale_factor = new_side_length / original_side_length

            #print(f'-> zoom in by {1/scale_factor}')
            #print(f"Img: {original_side_length}px to {new_side_length}px")
            target = target.copy()

            new_box = torch.zeros(4, dtype=torch.float32)
            new_box[0] = max(0, selected_box[0] - left)
            new_box[1] = max(0, selected_box[1] - top)
            new_box[2] = min(new_side_length, selected_box[2] - left)
            new_box[3] = min(new_side_length, selected_box[3] - top)

            # Ensure the new box is within the image boundaries
            new_box[2] = max(new_box[2], new_box[0] + 1)
            new_box[3] = max(new_box[3], new_box[1] + 1)

            return cropped_image, LabelSetDict(
                boxes=new_box.unsqueeze(0),
                labels=target.labels[box_index].unsqueeze(0)
            )

        return cropped_image, target

