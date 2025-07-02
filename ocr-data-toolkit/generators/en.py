from typing import List, Dict
from ..common.config import Config


class ENGenerator:
    def __init__(
        self, 
        num_samples: int = Config.num_samples, 
        bag_of_words: List[str] = Config.bag_of_words,
        text_probs: Dict[str, float] = Config.text_probs,
    ):
        self.num_samples = num_samples
        self.bag_of_words = bag_of_words
        self.text_probs = text_probs
    

    def __call__(self):
        ...
