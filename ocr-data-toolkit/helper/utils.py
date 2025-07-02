from typing import List, Tuple
from random import randint
import string
from PIL import Image, ImageOps, ImageChops, ImageFilter, ImageFont, ImageDraw
from datetime import datetime, timedelta
import math
import random
import cv2
import numpy as np


def resize_and_pad_image(img, output_image_size):
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
    return target

def generate_random_date():
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

def get_pil_font(font_list, font_size=22, font_weights=None):
    # Randomly select a font from a list of common fonts
    font_path = random.choices(font_list, font_weights, k=1)[0]
    font = ImageFont.truetype(font_path, font_size)    
    return font, font_path

def add_background(size, backgrounds: List[str]):
    index_random = random.randint(0, len(backgrounds) - 1)
    img = Image.open(backgrounds[index_random])
    img = img.resize(size)
    # draw = ImageDraw.Draw(img)
    return img

def apply_motion_blur(image, kernel_size=15, orientation=0):
    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    if orientation == 0:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_bokeh_blur(image, kernel_size=15, radius=7):
    # Create the bokeh kernel
    """Create a circular bokeh kernel."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if (i - center)**2 + (j - center)**2 <= radius**2:
                kernel[i, j] = 1
    kernel /= np.sum(kernel)

    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def getTwoLined(text):
    words = text.split()
    if len(words) < 2:
        return text
    split_word = words[random.randint(1, len(words)-1)]
    strs = sorted([w.strip() for w in text.split(split_word) if len(w) > 0], key=lambda x: len(x), reverse=True)
    return '\n'.join(strs) if len(strs) > 1 else strs[0]

def add_moire_patterns(image, alpha=0.2):
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

def guassianBlur(img, *kwargs):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1)))

def motionBlur(img, *kwargs):
    return Image.fromarray(
        apply_motion_blur(
            np.array(img, dtype=np.uint8), 
            kernel_size=random.randint(3, 6), 
            orientation=random.randint(1, 2)
        ))
    
def bokenBlur(img, *kwargs):
    if kwargs:
        img_w = kwargs[0]        
    return Image.fromarray(
        apply_bokeh_blur(
            np.array(img, dtype=np.uint8), 
            kernel_size=random.randint(3, 6), 
            radius=int(random.uniform(0.02, 0.6) * img_w)
        ))

