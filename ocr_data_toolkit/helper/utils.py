from typing import List, Tuple
import string
from PIL import Image, ImageFont
from datetime import datetime, timedelta
import random
import cv2
import numpy as np
import os
import re
import logging
from contextlib import contextmanager

@contextmanager
def silence_logger(logger_name: str, level=logging.WARNING):
    """
    Context manager to temporarily set the logging level for a specific logger.

    Args:
        logger_name (str): Name of the logger to silence.
        level (int, optional): Logging level to set temporarily (default: logging.WARNING).

    Usage:
        with silence_logger('my_logger', logging.ERROR):
            ...
    """
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)

def resize_and_pad_image(pil_img: Image, output_image_size: Tuple[int, int]) -> Image:
    """
    Resize a PIL image and pad it to the target output size, preserving aspect ratio.

    Args:
        pil_img (Image): Input PIL image.
        output_image_size (Tuple[int, int]): (width, height) of the output image.

    Returns:
        Image: The resized and padded PIL image.
    """
    # Convert PIL image to numpy array
    
    img = np.array(pil_img, dtype=np.uint8)
    input_width, input_height = output_image_size[:2]
    h, w, c = img.shape
    ratio = w / h
    resized_w = int(input_height * ratio)
    # Resize the width if it exceeds the input width
    isBottomPad = False
    resized_h = input_height
    if resized_w > input_width:
        resized_w = input_width
        resized_h = int(h / w * input_width)
        isBottomPad = True
    # Resize image while preserving aspect ratio
    img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    # Prepare target tensor with appropriate dimensions
    target = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    # Place the resized image into the target tensor
    if isBottomPad:
        target[:resized_h, :, :] = img
    else:
        target[:, :resized_w, :] = img
    return Image.fromarray(target)

def get_max_char_dimensions(font: ImageFont.ImageFont) -> Tuple[int, int]:
    """
    Get the maximum width and height of all printable ASCII characters for a given font.

    Args:
        font (ImageFont.ImageFont): The font to measure.

    Returns:
        Tuple[int, int]: (max_width, max_height) of all printable characters.
    """
    # Iterate over all printable ASCII characters
    
    max_width = 0
    max_height = 0
    # Use all printable ASCII characters (can be extended)
    characters = string.printable
    for char in characters:
        bbox = font.getbbox(char)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    return max_width, max_height

def generate_random_date():
    """
    Generate a random date string in a variety of common formats.

    Returns:
        str: Randomly formatted date string.
    """
    # Generate a random year, month, and day
    
    # Generate a random year, month, and day
    year = random.randint(1900, 2100)
    # month = random.randint(1, 12)
    month = random.choices(
        list(range(1, 13)),
        weights=[4 if v in [1, 4, 11] else 1 for v in range(1, 13)],
        k=1
    )[0]
    
    # Generate a random day based on the selected month (and considering leap years)
    max_day = (datetime(year, month % 12 + 1, 1) - timedelta(days=1)).day
    # day = random.randint(1, max_day)
    day = random.choices(
        list(range(1, max_day + 1)),
        weights=[4 if v in [1, 4, 11, 14, 21, 24] else 1 for v in range(1, max_day + 1)],
        k=1
    )[0]
    
    # Create a datetime object with the generated date
    date_obj = datetime(year, month, day)
    # Choose different date formats
    formats = dict({
        "%Y-%m-%d": 0.1,    # YYYY-MM-DD
        "%d-%m-%Y": 0.1,    # DD-MM-YYYY
        "%Y/%m/%d": 0.1,    # YYYY/MM/DD
        "%Y/%m/%d": 0.1,    # YYYY/MM/DD
        "%Y/%m/%d": 0.1,    # YYYY/MM/DD
        "%d/%m/%Y": 0.1,    # DD/MM/YYYY
        "%d/%m/%Y": 0.1,    # DD/MM/YYYY
        "%d/%m/%Y": 0.1,    # DD/MM/YYYY
        "%m/%d/%Y": 0.1,    # MM/DD/YYYY
        "%m/%d/%Y": 0.1,    # MM/DD/YYYY
        "%m/%d/%Y": 0.1,    # MM/DD/YYYY
        "%d.%m.%Y": 0.3,    # DD.MM.YYYY
        "%d %m %Y": 0.3,    # DD MM YYYY
        "%b %d, %Y": 0.5,   # Abbreviated month, day, year (e.g., Jan 13, 2023)
        "%d %b %Y": 0.5,   # Abbreviated month, day, year (e.g., 13 Jan 2023)
        "%d %b/%b %Y": 0.5,   # Abbreviated month, day, year (e.g., 13 Jan/Jan 2023)
        "%B %d, %Y": 0.2,    # Full month name, day, year (e.g., January 13, 2023)
        "%d %B %Y": 0.2,    # Full month name, day, year (e.g., January 13, 2023)
    })
    # Choose a random format
    date_format = random.choices(list(formats.keys()), weights=list(formats.values()), k=1)[0]

    date_ = date_obj.strftime(date_format)
    if date_format == "%d %b/%b %Y":
        elems = date_.split()
        d = elems[0]
        m = elems[1].split("/")[0]
        y = elems[2]
        r = "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 4)))
        m1 = random.choice([m, r])
        m2 = m if m1 == r else r
        date_ = f"{d} {m1}/{m2} {y}"

    if date_format in ["%b %d, %Y", "%d %b %Y", "%d %b/%b %Y"]:
        date_ = date_.upper()
    # Return the formatted date string
    return date_

def get_pil_font(font_list, font_size=32, font_weights=None):
    """
    Randomly select a font from a list and load it at the specified size.

    Args:
        font_list (List[str]): List of font file paths.
        font_size (int, optional): Font size to load.
        font_weights (List[float], optional): Weights for random selection.

    Returns:
        Tuple[ImageFont.FreeTypeFont, str]: The loaded font object and its file path.
    """
    # Randomly select a font from the list
    
    # Randomly select a font from a list of common fonts
    font_path = random.choices(font_list, font_weights, k=1)[0]
    font = ImageFont.truetype(font_path, font_size)    
    return font, font_path

def add_background(size, backgrounds: List[str]):
    """
    Select a random background image, resize it, and return as a PIL image.

    Args:
        size (Tuple[int, int]): Target image size (width, height).
        backgrounds (List[str]): List of background image file paths.

    Returns:
        Image: PIL image with the selected background.
    """
    # Randomly select and resize a background image
    
    index_random = random.randint(0, len(backgrounds) - 1)
    img = Image.open(backgrounds[index_random])
    img = img.resize(size)
    # draw = ImageDraw.Draw(img)
    return img

def getTwoLined(text):
    """
    Split a string into two lines at a random word boundary.

    Args:
        text (str): Input text.

    Returns:
        str: Text split into two lines (if possible).
    """
    # Randomly split text at a word boundary
    
    words = text.split()
    if len(words) < 2:
        return text
    split_word = words[random.randint(1, len(words)-1)]
    strs = sorted([w.strip() for w in text.split(split_word) if len(w) > 0], key=lambda x: len(x), reverse=True)
    return '\n'.join(strs) if len(strs) > 1 else strs[0]

def get_incremental_path(base_path: str, exp_name: str) -> str:
    """
    Returns an incremented experiment path based on existing folders.
    
    Example:
        If exp_name = "abc" and folders "abc", "abc1", "abc2" exist,
        it returns "abc3".
    
    Args:
        base_path (str): Directory in which to create the experiment folder.
        exp_name (str): Base name for the experiment folder.
    
    Returns:
        str: New experiment folder path with incremented suffix if needed.
    """
    # Find the next available experiment directory name
    
    """
    Returns an incremented experiment path based on existing folders.
    
    Example:
        If exp_name = "abc" and folders "abc", "abc1", "abc2" exist,
        it returns "abc3".
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    pattern = re.compile(rf'^{re.escape(exp_name)}(\d*)$')
    matches = [pattern.match(name) for name in existing]
    
    used_indices = []
    for m in matches:
        if m:
            index_str = m.group(1)
            used_indices.append(int(index_str) if index_str else 0)

    next_index = (max(used_indices) + 1) if used_indices else 0
    new_name = exp_name if next_index == 0 else f"{exp_name}{next_index}"
    return os.path.join(base_path, new_name)