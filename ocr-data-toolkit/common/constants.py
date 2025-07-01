from typing import List, Tuple, Union, Dict

SUPPORTED_LANGUAGES: List[str] = [
    'english', 'urdu', 'arabic'
]

TEXT_PROBS: Dict[str, float] = dict({
    "text": 0.7,
    "date": 0.1,
    "number": 0.2
})


BACKGROUNDS_PATH: str = 'data/backgrounds'
MRZS_PATH: str = "data/mrzs.txt"
ENGLISH_FONTS_PATH: str = "data/fonts/en_fonts"
URDU_FONTS_PATH: str = "data/fonts/urdu_fonts"
ARABIC_FONTS_PATH: str = "data/fonts/arabic_fonts"


