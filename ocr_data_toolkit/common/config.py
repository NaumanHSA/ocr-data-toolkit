import os
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict
from ..data import __data_path__
# from .. import __root_path__

__root_path__ = os.getcwd()

@dataclass
class Config:
    num_samples: int = 100
    supported_languages: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "en": {
            "fonts_path": os.path.join(__data_path__, "fonts/en"),
            "words_path": os.path.join(__data_path__, "vocab/en.txt")
        },
        # "urdu": {
        #     "fonts_path": os.path.join(__data_path__, "fonts/urdu"),
        #     "words_path": os.path.join(__data_path__, "vocab/urdu.txt")
        # },
        # "ar": {
        #     "fonts_path": os.path.join(__data_path__, "fonts/ar"),
        #     "words_path": os.path.join(__data_path__, "vocab/ar.txt")
        # },
        # "mrz": {
        #     "fonts_path": os.path.join(__data_path__, "fonts/mrz"),
        #     "words_path": os.path.join(__data_path__, "vocab/mrz.txt")
        # }
    })
    text_probs: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.7,
        "date": 0.1,
        "number": 0.2
    })
    language: str = None
    backgrounds_path: str = os.path.join(__data_path__, "backgrounds")
    fonts_path: str = None
    bag_of_words: List[str] = None    
    output_image_size: Tuple[int, int] = None
    split_train_test: bool = True
    train_test_ratio: float = 0.2
    output_save_path: str = os.path.join(__root_path__, "export")
    generate_mrz: bool = False


class AugmentationConfig:
    def __init__(self, user_config: Dict = None):
        default = {
            "max_num_words": 5,
            "num_lines": 1,
            "font_size": 36,
            "text_colors": ["#2f2f2f", "black", "#404040"],
            "letter_spacing_prob": 0.4,
            "margin_x": (0.5, 1.5),   # add padding to the image by margin (margin_x * character_width)
            "margin_y": (0.5, 1.5),   # add padding to the image by margin (margin_y * character_height)
            "blur_probs": {
                "gaussian": 0.3,
                "custom_blurs": 0.6
            },
            "moire_prob": 0.3,
            "opacity_prob": 0.3,
            "opacity_range": (150, 210),
            "brightness_range": (0.7, 1.2),
            "perspective_transform_prob": 0.3,
            "random_crop_width_range": (0.008, 0.01),
            "random_crop_height_range": (0.01, 0.1),
            "random_resize_factor_range": (0.9, 1.0),
            "random_stretch_factor_range": (0.1, 0.3),
        }
        self.config = {**default, **(user_config or {})}

    def __getitem__(self, key):
        return self.config[key]