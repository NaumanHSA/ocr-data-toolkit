# import packages
from typing import List, Tuple, Union, Dict
import cv2
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_curve
import skimage
from skimage.filters import try_all_threshold, threshold_otsu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cdist
import os
import requests
import sys
import random
import itertools
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
import glob
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, message_format='%(asctime)s - %(levelname)s - %(message)s')


from .common.constants import (
    SUPPORTED_LANGUAGES,
    TEXT_PROBS,
    BACKGROUNDS_PATH,
    MRZS_PATH,
    ENGLISH_FONTS_PATH,
    URDU_FONTS_PATH,
    ARABIC_FONTS_PATH
)


class ODT:
    def __init__(
        self,
        languages: List[str] = ['english'],
        backgrounds_path: str = BACKGROUNDS_PATH,
        fonts_path: str = None,
        mrz_path: str = MRZS_PATH,
        max_num_words: int = 10,
        text_probs: List[float] = TEXT_PROBS,
        output_image_size: Tuple[int, int] = (1024, 768),
        split_train_test: bool = True,
        train_test_ratio: Tuple[float, float] = (0.8, 0.2),
        output_save_path: str = None,
        generate_mrz: bool = False,
        logger: logging.RootLogger = logging.getLogger(__name__)
    ):
        self.languages = languages
        self.backgrounds_path = backgrounds_path
        self.fonts_path = fonts_path
        self.mrz_path = mrz_path
        self.max_num_words = max_num_words
        self.text_probs = text_probs
        self.output_image_size = output_image_size
        self.split_train_test = split_train_test
        self.train_test_ratio = train_test_ratio
        self.output_save_path = output_save_path
        self.generate_mrz = generate_mrz
        self.logger = logger
        self.__setup()
    

    def __setup(self):
        self.backgrounds: List[str] = []
        for bg_name in os.listdir(self.backgrounds_path):
            basename, ext = os.path.splitext(bg_name)
            if ext in ['.jpg', '.png', '.jpeg']:
                self.backgrounds.append(os.path.join(self.backgrounds_path, bg_name))

        self.logger.info("Total backgrounds found: %s", len(self.backgrounds))

        # download a list of words for use as background text
        word_site = "https://www.mit.edu/~ecprice/wordlist.100000"
        response = requests.get(word_site)
        self.bag_of_words = [x.decode() for x in response.content.splitlines()]

        # list of font types to exclude
        self.fonts = []
        self.fonts_weights = []
        for font_path in glob.glob(os.path.join(self.fonts_path, "**", "*.ttf"), recursive=True):
            self.fonts.append(font_path)
            if "Passport" in font_path:
                self.fonts_weights.append(5)
            elif "Visa" in font_path:
                self.fonts_weights.append(1)
            else:
                self.fonts_weights.append(2)

        self.mrz = []
        with open(self.mrz_path, "r") as reader:
            self.mrz = [r.replace("\n", "").strip() for r in reader.readlines()]

        self.mrz.extend(generate_mrz_list(2000))
        self.punctuations = ['-', '<', '/', ',', "'", ':', '&', '.', '(', ')']