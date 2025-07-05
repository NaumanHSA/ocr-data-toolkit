from typing import List, Dict, Tuple
import string
import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageEnhance
from datetime import datetime, timedelta
from collections import Counter
import random
import cv2
import numpy as np
        
from ..common import AugmentationConfig


class Augmentation:
    """
    Image augmentation class for synthetic OCR data generation.

    Provides a suite of image augmentation methods such as noise, blur, perspective, elastic distortion, and more.
    All methods operate on PIL Images and are configurable via an AugmentationConfig object.
    """
    def __init__(
        self,
        augmentation_config: AugmentationConfig = None
    ):
        """
        Initialize the Augmentation class.

        Args:
            augmentation_config (AugmentationConfig, optional): Configuration for augmentations.
        """
        self.cfg = augmentation_config
    
    def add_noise(self, img):
        """
        Add salt-and-pepper noise to the image.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Noisy image.
        """
        # basic salt-and-pepper noise
        
        # basic salt-and-pepper noise
        arr = np.array(img)
        noise = np.random.randint(0, 2, arr.shape[:2]) * 255
        mask = np.random.rand(*arr.shape[:2]) > 0.98
        arr[mask] = noise[mask, None]
        return Image.fromarray(arr)

    def random_crop(self, img):
        """
        Randomly crop the image within configured width and height ranges.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Cropped image.
        """
        
        w, h = img.size
        x1 = random.uniform(*self.cfg['random_crop_width_range']) * w
        y1 = random.uniform(*self.cfg['random_crop_height_range']) * h
        x2 = w - (random.uniform(*self.cfg['random_crop_width_range']) * w)
        y2 = h - (random.uniform(*self.cfg['random_crop_height_range']) * h)
        return img.crop((x1, y1, x2, y2))

    def random_resize(self, img):
        """
        Randomly resize the image by a factor within a configured range.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Resized image (or original if not applied).
        """
        
        if random.random() > 0.7:
            resize_factor = random.uniform(*self.cfg['random_resize_factor_range'])
            w, h = img.size
            return img.resize((int(w * resize_factor), int(h * resize_factor)))
        return img

    def random_stretch(self, img):
        """
        Randomly stretch or compress the image height within a configured range.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Stretched image.
        """
        
        w, h = img.size
        stretch = random.choice([1 + random.uniform(*self.cfg['random_stretch_factor_range']), 1 - random.uniform(0.05, 0.2)])
        return img.resize((w, int(h * stretch)))

    def add_moire_patterns(self, image, alpha=0.2):
        """
        Overlay a synthetic moire pattern using sinusoidal waves.

        Args:
            image (Image.Image): Input PIL image.
            alpha (float, optional): Blending factor for the moire pattern.

        Returns:
            Image.Image: Image with moire pattern overlay.
        """
        
        """
        Generates a moire pattern using sinusoidal waves.
        Parameters:
        - height: Height of the pattern
        - width: Width of the pattern
        - frequency: Frequency of the sine wave
        - amplitude: Amplitude of the sine wave
        Returns:
        - Moire pattern as a 2D numpy array
        """
        image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # Initialize the RGB pattern with zeros
        pattern_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for channel in range(3):  # Iterate over R, G, B channels
            frequency = random.uniform(10, 30)
            amplitude = random.uniform(5, 15)
            angle = random.uniform(0, np.pi) * random.choice([-1, 1])

            x = np.arange(0, width)
            y = np.arange(0, height)
            X, Y = np.meshgrid(x, y)
            # Create a sinusoidal pattern with some phase shift
            pattern = amplitude * np.sin(2 * np.pi * frequency * X / width + 2 * angle * frequency * Y / height)
            # Normalize the final pattern to the range [0, 255] and assign to the respective RGB channel
            pattern_rgb[:, :, channel] = ((pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern)) * 255).astype(np.uint8)

        noisy_image = cv2.addWeighted(image, 1 - alpha, pattern_rgb, alpha, 0)
        return Image.fromarray(noisy_image)

    def guassianBlur(self, img, **kwargs):
        """
        Apply Gaussian blur to the image.

        Args:
            img (Image.Image): Input PIL image.
            **kwargs: Additional arguments (unused).

        Returns:
            Image.Image: Blurred image.
        """
        
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1)))

    def motionBlur(self, img, **kwargs):
        """
        Apply motion blur to the image using a random orientation and kernel size.

        Args:
            img (Image.Image): Input PIL image.
            **kwargs: Additional arguments (unused).

        Returns:
            Image.Image: Image with motion blur.
        """
        
        image = np.array(img, dtype=np.uint8)
        orientation=random.randint(1, 2)
        kernel_size=random.randint(3, 6)
        # Create a motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        if orientation == 0:
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        else:
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        # Apply the kernel to the image
        blurred = cv2.filter2D(image, -1, kernel)
        return Image.fromarray(blurred)
        
    def bokenBlur(self, img, **kwargs):
        """
        Apply bokeh (out-of-focus) blur to the image using a circular kernel.

        Args:
            img (Image.Image): Input PIL image.
            **kwargs: width (int): Image width for kernel scaling.

        Returns:
            Image.Image: Image with bokeh blur.
        """
        
        img_w = kwargs.get("width")
        # Create the bokeh kernel
        """Create a circular bokeh kernel."""
        image = np.array(img, dtype=np.uint8)
        kernel_size = random.randint(3, 6)
        radius = int(random.uniform(0.02, 0.6) * img_w)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - center)**2 + (j - center)**2 <= radius**2:
                    kernel[i, j] = 1
        kernel /= np.sum(kernel)
        # Apply the kernel to the image
        blurred = cv2.filter2D(image, -1, kernel)
        return Image.fromarray(blurred)

    def apply_perspective_transform(self, pil_img: Image.Image, max_warp: float = 0.15) -> Image.Image:
        """
        Apply a random perspective (projective) transformation to the image.

        Args:
            pil_img (Image.Image): Input PIL image.
            max_warp (float, optional): Maximum fraction for corner perturbation.

        Returns:
            Image.Image: Warped image.
        """
        
        img = np.array(pil_img)
        h, w = img.shape[:2]
        # Source points (corners)
        src = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        # Destination points with random perturbations
        dst = np.float32([
            [w * random.uniform(0, max_warp), h * random.uniform(0, max_warp)],
            [w * (1 - random.uniform(0, max_warp)), h * random.uniform(0, max_warp)],
            [w * (1 - random.uniform(0, max_warp)), h * (1 - random.uniform(0, max_warp))],
            [w * random.uniform(0, max_warp), h * (1 - random.uniform(0, max_warp))]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

    def elastic_distortion(self, pil_img: Image.Image, alpha=50, sigma=5) -> Image.Image:
        """
        Apply elastic distortion to the image (simulate handwriting or paper warping).

        Args:
            pil_img (Image.Image): Input PIL image.
            alpha (float, optional): Scaling factor for displacement.
            sigma (float, optional): Gaussian filter sigma.

        Returns:
            Image.Image: Distorted image.
        """
        
        img = np.array(pil_img.convert("L"))

        random_state = np.random.RandomState(None)
        shape = img.shape

        dx = (random_state.rand(*shape) * 2 - 1)
        dy = (random_state.rand(*shape) * 2 - 1)

        dx = cv2.GaussianBlur(dx, (17, 17), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (17, 17), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(distorted).convert("RGB")

    def add_ink_bleed(self, pil_img: Image.Image, radius=1, iterations=1) -> Image.Image:
        """
        Simulate ink bleed by applying a max filter multiple times.

        Args:
            pil_img (Image.Image): Input PIL image.
            radius (int, optional): Radius for the max filter.
            iterations (int, optional): Number of times to apply the filter.

        Returns:
            Image.Image: Image with simulated ink bleed.
        """
        
        img = pil_img.convert("L")
        for _ in range(iterations):
            img = img.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))
        return img.convert("RGB")

    def simulate_low_resolution(self, pil_img: Image.Image, scale: float = 0.5) -> Image.Image:
        """
        Simulate low resolution by downsampling and upsampling the image.

        Args:
            pil_img (Image.Image): Input PIL image.
            scale (float, optional): Downsampling scale factor.

        Returns:
            Image.Image: Image with simulated low resolution.
        """
        
        w, h = pil_img.size
        down = pil_img.resize((int(w * scale), int(h * scale)), resample=Image.BILINEAR)
        up = down.resize((w, h), resample=Image.BICUBIC)
        return up