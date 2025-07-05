# import packages
import os
from typing import List, Tuple, Union, Dict
import warnings
from threading import Thread
import matplotlib.pyplot as plt
from PIL import ImageOps, Image, ImageDraw, ImageFont
import glob
from datetime import datetime, timedelta
from atpbar import atpbar

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)

from .common.config import Config
from .generators import ENGenerator
from .helper.utils import resize_and_pad_image, silence_logger


class ODT:
    """
    Optical Character Recognition (OCR) Data Toolkit main class.

    This class provides an interface for generating synthetic OCR training data, including images and ground-truth labels, using configurable backgrounds, fonts, and text sources. It supports multiprocessing for efficient data generation and utilities for visualizing font catalogs.

    Attributes:
        language (str): Language code for data generation (default from Config).
        bag_of_words (List[str]): List of words to use for text synthesis.
        backgrounds_path (str): Path to background images directory.
        fonts_path (str): Path to fonts directory.
        text_probs (Dict[str, float]): Probabilities for text composition.
        output_image_size (Tuple[int, int]): Output image size (width, height).
        train_test_ratio (float): Ratio for splitting train/test data.
        output_save_path (str): Root path for saving generated data.
        augmentation_config (Dict): Augmentation configuration.
        num_workers (int): Number of worker threads for parallel generation.
        logger (logging.RootLogger): Logger for toolkit events.
        generator (ENGenerator): Text-image generator instance.
    """
    def __init__(
        self,
        language: str = Config.language,
        bag_of_words: List[str] = Config.bag_of_words,
        backgrounds_path: str = Config.backgrounds_path,
        fonts_path: str = Config.fonts_path,
        text_probs: Dict[str, float] = None,
        output_image_size: Tuple[int, int] = Config.output_image_size,
        train_test_ratio: float = Config.train_test_ratio,
        output_save_path: str = Config.output_save_path,
        augmentation_config: Dict = None,
        num_workers: int = 4,
        logger: logging.RootLogger = logging.getLogger(__name__)
    ):
        """
        Initialize the ODT class with configuration for synthetic data generation.

        Args:
            language (str): Language code for text generation.
            bag_of_words (List[str]): List of words to use for text.
            backgrounds_path (str): Directory containing background images.
            fonts_path (str): Directory containing font files.
            text_probs (Dict[str, float], optional): Probabilities for text composition.
            output_image_size (Tuple[int, int], optional): Output image size (width, height).
            train_test_ratio (float, optional): Ratio for train/test split.
            output_save_path (str, optional): Path to save generated data.
            augmentation_config (Dict, optional): Augmentation settings.
            num_workers (int, optional): Number of worker threads.
            logger (logging.RootLogger, optional): Logger for toolkit events.
        """
        self.language = language
        self.backgrounds_path = backgrounds_path
        self.fonts_path = fonts_path
        self.bag_of_words = bag_of_words
        self.text_probs = text_probs
        self.output_image_size = output_image_size
        self.train_test_ratio = train_test_ratio
        self.output_save_path = output_save_path
        self.logger = logger
        self.num_workers = num_workers
        self.augmentation_config = augmentation_config
        self.generator = None
        self.__setup()
    

    def __setup(self):
        """
        Internal setup for backgrounds, fonts, generator, and output paths.

        Loads backgrounds and fonts, validates language, sets up the text generator, and prepares output directories.
        Raises:
            ValueError: If language is not specified or not supported.
        """
        config = Config()
        if self.language is None:
            raise ValueError("Language must be specified")
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
        self.logger.info("Backgrounds loaded.")

        # list of font types to exclude
        if self.fonts_path is None:
            self.fonts_path = config.supported_languages[self.language]["fonts_path"]

        self.fonts = []
        for font_path in glob.glob(os.path.join(self.fonts_path, "**", "*.ttf"), recursive=True):
            self.fonts.append(font_path)
        self.logger.info("Fonts loaded.")

        if self.text_probs is None:
            self.text_probs = config.text_probs
        self.punctuations = ['-', '<', '/', ',', "'", ':', '&', '.', '(', ')']

        # paths
        self.output_save_path = os.path.join(self.output_save_path, f"{self.language}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.train_path = os.path.join(self.output_save_path, "train")
        self.test_path = os.path.join(self.output_save_path, "test")
        self.train_test_ratio = self.train_test_ratio if 0 < self.train_test_ratio < 1 else 0.2
        self.generator = ENGenerator(
            bag_of_words=self.bag_of_words,
            text_probs=self.text_probs,
            fonts=self.fonts,
            backgrounds=self.backgrounds,
            augmentation_config=self.augmentation_config
        )

    def generate_single_image(self):
        """
        Generate a single synthetic image and its corresponding text.

        Returns:
            Tuple[str, PIL.Image.Image]: The generated text and the image.
        """
        text, image = self.generator()
        # resize and pad image
        if self.output_image_size is not None:
            image = resize_and_pad_image(image, self.output_image_size)
        return text, image
    
    def __generate(self, pid, num_images, train_path, test_path, generator, test_split):
        """
        Internal method to generate images and labels for a specific worker.

        Args:
            pid (int): Worker/process ID.
            num_images (int): Number of images to generate.
            train_path (str): Directory for training images and labels.
            test_path (str): Directory for test images and labels.
            generator (callable): Generator function for text-image pairs.
            test_split (float): Fraction of images to allocate to test set.
        """
        train_writer = open(os.path.join(train_path, f"{pid}_gt.txt"), "w")
        test_writer = open(os.path.join(test_path, f"{pid}_gt.txt"), "w")
        for image_index in atpbar(range(num_images), name=f'Job {pid}'):
            text, img = generator()
            # resize and pad image
            if self.output_image_size is not None:
                img = resize_and_pad_image(img, self.output_image_size)
            safe_text = text.replace('\n', '\\n')
            image_name = f"{pid}_{image_index}_Synthetic_En.jpg"
            if image_index % int(1/test_split) == 0:
                # put in test
                image_path = os.path.join(test_path, "images", image_name)
                img.convert('RGB').save(image_path)
                test_writer.write(f"{image_name}\t{safe_text}\n")
            else:
                # put in train
                image_path = os.path.join(train_path, "images", image_name)
                img.convert('RGB').save(image_path)
                train_writer.write(f"{image_name}\t{safe_text}\n")
        train_writer.close()
        test_writer.close()


    def generate_training_data(self, num_samples: int):
        """
        Generate a full synthetic OCR dataset for training and testing.

        Args:
            num_samples (int): Total number of images to generate.
        """
        os.makedirs(os.path.join(self.test_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.train_path, "images"), exist_ok=True)
        num_images_per_process = int(num_samples / self.num_workers)
        remain_images = num_samples % self.num_workers
        self.logger.info("Generating images...")
        threads = [None] * self.num_workers
        for pid in range(self.num_workers):
            threads[pid] = Thread(
                target=self.__generate, 
                args=(
                    pid,
                    (num_images_per_process + remain_images) if pid == 0 else num_images_per_process,
                    self.train_path,
                    self.test_path,
                    self.generator,
                    self.train_test_ratio
                )
            )
            threads[pid].start()
        # join threads
        for thread in threads:
            thread.join()
            
        self.logger.info("Finalizing...")
        train_writer = open(os.path.join(self.train_path, "gt.txt"), "w")
        test_writer = open(os.path.join(self.test_path, "gt.txt"), "w")    
        train_records, test_records = 0, 0
        for path in [self.train_path, self.test_path]:
            for pid in range(self.num_workers):
                with open(os.path.join(path, f"{pid}_gt.txt"), "r") as reader:
                    for line in reader.readlines():
                        if "train" in path:
                            train_writer.write(line)
                            train_records += 1
                        else:
                            test_writer.write(line)
                            test_records += 1                    
        train_writer.close()
        test_writer.close()
        
        # remove other GTs
        for path in [self.train_path, self.test_path]:
            for pid in range(self.num_workers):
                os.remove(os.path.join(path, f"{pid}_gt.txt"))
                            
        self.logger.info(f"Number of Train records {train_records}")
        self.logger.info(f"Number of Test records {test_records}")


    def visualize_font_catalog(self, save_dir: str = "font_catalog", chunk_size: int = 10):
        """
        Visualize all available fonts by generating sample images for each font.

        Args:
            save_dir (str, optional): Directory to save font catalog images.
            chunk_size (int, optional): Number of fonts per catalog image.
        """
        with silence_logger("ocr_data_toolkit.odt"):   # silence matplotlib logger
            os.makedirs(save_dir, exist_ok=True)
            num_fonts = len(self.fonts)
            num_chunks = (num_fonts + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_fonts)
                fonts_subset = self.fonts[start:end]
                fig, axes = plt.subplots(len(fonts_subset), 1, figsize=(12, len(fonts_subset) * 2.5))
                # Ensure axes is iterable
                if len(fonts_subset) == 1:
                    axes = [axes]
                for i, font_path in enumerate(fonts_subset):
                    text, img = self.generate_single_image()
                    if self.output_image_size is not None:
                        img = resize_and_pad_image(img, self.output_image_size)
                    axes[i].imshow(img)
                    font_name = os.path.basename(font_path)
                    title_text = f"Font: {font_name}\nText:\n{text}"
                    axes[i].text(
                        0, 1.05, title_text, fontsize=10,
                        transform=axes[i].transAxes, ha='left', va='bottom'
                    )
                    axes[i].axis("off")
                plt.tight_layout(h_pad=5.0)  # vertical spacing between images
                save_path = os.path.join(save_dir, f"font_catalog_{chunk_idx}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

        self.logger.info(f"✅ Font catalog images saved in: {os.path.abspath(save_dir)}")
