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
logging.basicConfig(level=logging.INFO)

from common.config import Config
from generators.base import GeneratorBase



class ODT:
    def __init__(
        self,
        num_samples: int = Config.num_samples,
        language: str = Config.language,
        max_num_words: int = Config.max_num_words,
        bag_of_words: List[str] = Config.bag_of_words,
        backgrounds_path: str = Config.backgrounds_path,
        fonts_path: str = Config.fonts_path,
        text_probs: Dict[str, float] = None,
        output_image_size: Tuple[int, int] = Config.output_image_size,
        split_train_test: bool = Config.split_train_test,
        train_test_ratio: Tuple[float, float] = Config.train_test_ratio,
        output_save_path: str = Config.output_save_path,
        logger: logging.RootLogger = logging.getLogger(__name__)
    ):
        self.num_samples = num_samples
        self.language = language
        self.backgrounds_path = backgrounds_path
        self.fonts_path = fonts_path
        self.max_num_words = max_num_words
        self.bag_of_words = bag_of_words
        self.text_probs = text_probs
        self.output_image_size = output_image_size
        self.split_train_test = split_train_test
        self.train_test_ratio = train_test_ratio
        self.output_save_path = output_save_path
        self.logger = logger

        self.generator = None
        self.__setup()
    

    def __setup(self):
        config = Config()
        if self.language not in config.supported_languages:
            raise ValueError(f"Language {self.language} is not supported. Supported languages are {config.supported_languages}")

        if self.bag_of_words is None:
            self.words_path = config.supported_languages[self.language]["words_path"]
            self.bag_of_words = [x.replace("\n", "").strip() for x in open(self.words_path, "r").readlines() if x.strip() != ""][1:]
        
        self.backgrounds: List[str] = []
        for bg_name in os.listdir(self.backgrounds_path):
            basename, ext = os.path.splitext(bg_name)
            if ext in ['.jpg', '.png', '.jpeg']:
                self.backgrounds.append(os.path.join(self.backgrounds_path, bg_name))
        self.logger.info("Total backgrounds found: %s", len(self.backgrounds))

        # list of font types to exclude
        if self.fonts_path is None:
            self.fonts_path = config.supported_languages[self.language]["fonts_path"]

        self.fonts = []
        for font_path in glob.glob(os.path.join(self.fonts_path, "**", "*.ttf"), recursive=True):
            self.fonts.append(font_path)

        if self.text_probs is None:
            self.text_probs = config.text_probs
        self.punctuations = ['-', '<', '/', ',', "'", ':', '&', '.', '(', ')']

    def generate(self):
        gen = GeneratorBase(
            num_samples=self.num_samples,
            bag_of_words=self.bag_of_words,
            text_probs=self.text_probs,
            fonts=self.fonts,
            backgrounds=self.backgrounds,
        )
        text, image = gen.generate()

        print(text)

        image.save("test.png")
    

