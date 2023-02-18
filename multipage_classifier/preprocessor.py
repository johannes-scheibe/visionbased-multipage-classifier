from typing import Dict
from torchvision.transforms.functional import resize, rotate
from torchvision import transforms
from PIL import ImageOps, Image
import numpy as np
import torch

class ImageProcessor:
    def __init__(self, img_size:tuple = (224,224)):
        self.input_size = img_size
        self.align_long_axis = False
        self.random_padding = False

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def prepare_input(self, img: Image.Image, random_padding: bool = False) -> torch.Tensor:

        img = img.convert("RGB")
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.height > img.width)
            or (self.input_size[0] < self.input_size[1] and img.height < img.width)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[0], self.input_size[1]))
        delta_width = self.input_size[0] - img.width
        delta_height = self.input_size[1] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        pixel_values = torch.Tensor(self.to_tensor(ImageOps.expand(img, padding)))
        return pixel_values