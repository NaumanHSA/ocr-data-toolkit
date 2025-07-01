from typing import List, Tuple, Union, Dict

from .constants import (
    SUPPORTED_LANGUAGES,
    TEXT_PROBS,
    BACKGROUNDS_PATH,
    MRZS_PATH,
    ENGLISH_FONTS_PATH,
    URDU_FONTS_PATH,
    ARABIC_FONTS_PATH
)


config: Dict = {
    "languages": ['english'],
    "backgrounds_path": BACKGROUNDS_PATH,
    "fonts_path": None,
    "mrz_path": MRZS_PATH,
    "max_num_words": 10,
    "text_probs": TEXT_PROBS,
    "output_image_size": (1024, 768),
    "split_train_test": True,
    "train_test_ratio": (0.8, 0.2),
    "output_save_path": None,
    "generate_mrz": False,
}
