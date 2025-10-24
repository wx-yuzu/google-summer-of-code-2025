from typing import Tuple, Union

import cv2
import torch
import torchvision.transforms as T
from PIL.Image import Image


class PreProcessor:
    """Prepare images for model inference by resizing and tensorizing.

    Parameters
    ----------
    image_size : tuple[int, int]
        Target size ``(height, width)`` used for resizing input images.
    """

    def __init__(self, image_size: Tuple[int, int]):
        self.resize_image_size = image_size

    # [TODO] Make this class callable (implement __call__ delegating to preprocess_image).
    def preprocess_image(self, image: Union[cv2.Mat, Image]) -> torch.Tensor:
        """Resize an image and convert it to a batched torch tensor.

        The image is resized to ``self.resize_image_size`` and converted to a
        float tensor in ``[0, 1]`` with shape ``(1, C, H, W)`` (a batch dimension
        is added). Channel order considerations:
        - PIL images are assumed RGB.
        - OpenCV matrices (``cv2.Mat``) are typically BGR; if color fidelity
          matters, convert to RGB before calling or adjust downstream.

        Parameters
        ----------
        image : cv2.Mat | PIL.Image.Image
            Input image to preprocess.

        Returns
        -------
        torch.Tensor
            A 4D tensor of shape ``(1, C, H, W)`` suitable for model input.
        """
        transform = T.Compose(
            [
                T.Resize(self.resize_image_size),
                T.ToTensor(),
            ]
        )
        return transform(image).unsqueeze(0)  # Add batch dim
