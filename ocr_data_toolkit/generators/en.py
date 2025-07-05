from typing import List, Dict, Tuple
import string
import random
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageEnhance

from ..common import AugmentationConfig
from ..helper.utils import (
    get_pil_font,
    add_background,
    generate_random_date,
    get_max_char_dimensions
)
from ..helper.augmentation import Augmentation


class ENGenerator:
    """
    English Text-Image Generator for Synthetic OCR Data.

    This class generates synthetic text samples (random words, dates, numbers) and renders them onto images with configurable fonts, backgrounds, and augmentations. It is used for creating training data for OCR models.

    Attributes:
        bag_of_words (List[str]): List of words for text synthesis.
        text_probs (Dict[str, float]): Probabilities for text, date, or number generation.
        fonts (List[str]): List of font file paths.
        backgrounds (List[str]): List of background image file paths.
        punctuations (List[str]): List of punctuation characters for text augmentation.
        cfg (AugmentationConfig): Augmentation configuration object.
        augmentation (Augmentation): Augmentation operations.
    """
    def __init__(
        self,
        bag_of_words: List[str],
        text_probs: Dict[str, float],
        fonts: List[str],
        backgrounds: List[str],
        augmentation_config: Dict = None
    ):
        """
        Initialize the ENGenerator.

        Args:
            bag_of_words (List[str]): List of words for generating random text.
            text_probs (Dict[str, float]): Probabilities for generating text, date, or number.
            fonts (List[str]): List of font file paths.
            backgrounds (List[str]): List of background image file paths.
            augmentation_config (Dict, optional): Augmentation configuration.
        """
        self.bag_of_words = bag_of_words
        self.text_probs = text_probs
        self.fonts = fonts
        self.backgrounds = backgrounds
        self.punctuations = ['-', '<', '/', ',', "'", ':', '&', '.', '(', ')']
        self.cfg = AugmentationConfig(augmentation_config)
        self.augmentation = Augmentation(self.cfg)
    
    def _generate_text(self) -> str:
        """
        Generate a random text sample, date, or number based on configured probabilities.

        Returns:
            str: Generated text sample.
        """
        # Choose what type of text to generate (text/date/number) based on probabilities
        
        toGenerate = random.choices(list(self.text_probs.keys()), weights=self.text_probs.values(), k=1)[0]
        if toGenerate == "text":
            num_lines = random.randint(*self.cfg["num_lines"]) if isinstance(self.cfg["num_lines"], tuple) else self.cfg["num_lines"]
            total_words = random.randint(self.cfg["max_num_words"] * (num_lines), self.cfg["max_num_words"] * num_lines)
            words = random.choices(self.bag_of_words, k=total_words)
            lines = []
            words_per_line = total_words // num_lines
            for i in range(num_lines):
                if i == num_lines - 1:
                    # Last line is shorter
                    line_words = words[i * words_per_line : ]
                    if len(line_words) > 4:
                        line_words = line_words[:len(line_words) // 2]  # cut it short
                else:
                    line_words = words[i * words_per_line : (i + 1) * words_per_line]
                line = ' '.join(line_words)
                # Add optional punctuation
                if random.random() > 0.7 and len(line) > 3:
                    punct = random.choice(self.punctuations)
                    if punct != '<':
                        si = random.choice([*[i for i, x in enumerate(line) if x == ' '], 0, len(line) - 1])
                        ps = random.choice([f' {punct}', f' {punct} ', f'{punct} '])
                        line = line[:si] + ps + line[si + 1:]
                lines.append(line)
                
            to_case = random.choices([str.upper, str.lower, str.capitalize, str.title], weights=[0.3, 0.3, 0.2, 0.2], k=1)[0]
            text = to_case("\n".join(lines))

        elif toGenerate == "date":
            text = self._generate_date()
        else:
            text = self._generate_number()
        return text
    
    def _generate_date(self) -> str:
        """
        Generate a random date string using the helper function.

        Returns:
            str: Randomly generated date string.
        """
        return generate_random_date()

    def _generate_number(self) -> str:
        """
        Generate a random alphanumeric number string.

        Returns:
            str: Randomly generated number string.
        """
        # Build a string of random digits and uppercase letters, sometimes inserting a dash
        
        l = []
        for _ in range(random.randint(1, 15)):
            if random.random() > 0.3:
                letter = string.digits[random.randint(0, 9)]
            else:
                letter = random.choice([string.ascii_uppercase[random.randint(0, 25)], "-"])
            l.append(letter)
        return ''.join(l)

    def _estimate_image_size(self, text: str, font: ImageFont.ImageFont, letter_spacing: int) -> Tuple[int, int, int, int, int]:
        """
        Estimate the required image size for rendering the given text with the specified font and letter spacing.

        Args:
            text (str): Text to render.
            font (ImageFont.ImageFont): Font to use.
            letter_spacing (int): Extra spacing between letters.

        Returns:
            Tuple[int, int, int, int, int]: (final_width, final_height, margin_x, margin_y)
        """
        # Compute the width and height required to render the text, including margins
        
        lines = text.split("\n")
        img_w = 0
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        for line in lines:
            line_width = 0
            for char in line:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                line_width += (char_width + letter_spacing)
            img_w = max(img_w, line_width)
        img_h = line_height * len(lines)

        max_char_width, max_char_height = get_max_char_dimensions(font)
        margin_x = int(random.uniform(*self.cfg["margin_x"]) * max_char_width)
        margin_y = int(random.uniform(*self.cfg["margin_y"]) * max_char_height)
        final_width = img_w + margin_x
        final_height = img_h + margin_y
        return final_width, final_height, margin_x, margin_y

    def _generate_single_image(self, text: str = None) -> Tuple[str, Image.Image]:
        """
        Generate a single synthetic image and its corresponding text.

        Args:
            text (str, optional): Text to render. If None, generates random text.

        Returns:
            Tuple[str, Image.Image]: The generated text and the rendered image.
        """
        # Select font and generate text if not provided
        
        font, font_path = get_pil_font(self.fonts, font_size=self.cfg['font_size'])
        if text is None:
            text = self._generate_text()
        letter_spacing = random.randint(1, 4) if random.random() < self.cfg['letter_spacing_prob'] else 0
        final_width, final_height, margin_x, margin_y = self._estimate_image_size(text, font, letter_spacing)
        img = self._create_base_image(final_width, final_height)
        self._draw_text(img, text, font, letter_spacing, margin_x, margin_y)
        img = self._apply_postprocessing(img, final_width)
        clean_text = ' '.join(text.split(' '))
        return clean_text, img.convert("RGB")

    def _create_base_image(self, width: int, height: int) -> Image.Image:
        """
        Create a base image, either with a random background or plain white.

        Args:
            width (int): Image width.
            height (int): Image height.

        Returns:
            Image.Image: The base image.
        """
        # With 80% probability, use a random background; otherwise, use a white canvas
        
        if random.random() > 0.2:
            return add_background((width, height), self.backgrounds)
        else:
            return Image.new('L', (width, height), color='white')

    def _draw_text(self, img: Image.Image, text: str, font: ImageFont.ImageFont, letter_spacing: int, margin_x: int, margin_y: int):
        """
        Draw the provided text onto the image with the specified font and spacing.

        Args:
            img (Image.Image): Image to draw on.
            text (str): Text to render.
            font (ImageFont.ImageFont): Font to use.
            letter_spacing (int): Space between letters.
            margin_x (int): Horizontal margin.
            margin_y (int): Vertical margin.
        """
        # Draw each character of each line, handling random x/y offset and color
        
        draw = ImageDraw.Draw(img)
        x_init = random.randint(0, margin_x)
        y = margin_y // 2
        color = random.choice(self.cfg['text_colors'])
        for line in text.split('\n'):
            x = x_init
            for char in line:
                bbox = draw.textbbox((0, 0), char, font=font)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                y = min(max(y, margin_y // 2), img.height - margin_y // 2 - char_height)
                draw.text((x, y), char, fill=color, font=font, align="center")
                x += char_width + letter_spacing
            y += int(char_height * 1.6)

    def _apply_postprocessing(self, img: Image.Image, width: int) -> Image.Image:
        """
        Apply postprocessing augmentations to the image (blur, brightness, moire, perspective, etc).

        Args:
            img (Image.Image): Image to process.
            width (int): Image width (for some augmentations).

        Returns:
            Image.Image: Augmented image.
        """
        # Apply a series of random augmentations to simulate real-world distortions
        
        img = img.convert("RGB")
        if random.random() < self.cfg['blur_probs']['gaussian']:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.5)))

        if random.random() < self.cfg['blur_probs']['custom_blurs']:
            for op in random.choices([
                self.augmentation.guassianBlur,
                self.augmentation.motionBlur,
                self.augmentation.bokenBlur
            ], k=2):
                img = op(img, width=width)

        if random.random() < self.cfg['opacity_prob']:
            img.putalpha(random.randint(*self.cfg['opacity_range']))

        if random.random() < self.cfg['moire_prob']:
            img = self.augmentation.add_moire_patterns(img, alpha=random.uniform(0.1, 0.3))

        brightness_prob = 0.5
        if random.random() < brightness_prob:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(*self.cfg['brightness_range']))

        if random.random() < self.cfg['perspective_transform_prob']:
            img = self.augmentation.apply_perspective_transform(img)
        return img

    def __call__(self, text: str = None) -> Tuple[str, Image.Image]:
        """
        Callable interface: generate a single text-image pair.

        Args:
            text (str, optional): Text to render. If None, generates random text.

        Returns:
            Tuple[str, Image.Image]: The generated text and image.
        """
        return self._generate_single_image(text)

    
