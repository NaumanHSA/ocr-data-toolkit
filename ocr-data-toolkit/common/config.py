from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict


@dataclass
class Config:
    num_samples: int = 100
    supported_languages: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "en": {
            "fonts_path": "data/fonts/en",
            "words_path": "data/vocab/en.txt"
        },
        "urdu": {
            "fonts_path": "data/fonts/urdu",
            "words_path": "data/vocab/urdu.txt"
        },
        "ar": {
            "fonts_path": "data/fonts/ar",
            "words_path": "data/vocab/ar.txt"
        },
        "mrz": {
            "fonts_path": "data/fonts/mrz",
            "words_path": "data/vocab/mrz.txt"
        }
    })
    text_probs: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.7,
        "date": 0.1,
        "number": 0.2
    })
    language: str = None
    backgrounds_path: str = 'data/backgrounds'
    fonts_path: str = None
    max_num_words: int = 10
    bag_of_words: List[str] = None    
    output_image_size: Tuple[int, int] = None
    split_train_test: bool = True
    train_test_ratio: Tuple[float, float] = (0.8, 0.2)
    output_save_path: str = None
    generate_mrz: bool = False


