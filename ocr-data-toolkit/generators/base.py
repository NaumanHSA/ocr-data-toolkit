from typing import List, Dict, Tuple
import string
import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageEnhance
from datetime import datetime, timedelta
from collections import Counter
import random
from collections import Counter

from common.config import Config
from helper.utils import (
    get_pil_font,
    add_background,
    guassianBlur,
    motionBlur,
    bokenBlur,
    add_moire_patterns,
    generate_random_date,
)


class GeneratorBase:
    def __init__(
        self,
        num_samples: int,
        bag_of_words: List[str],
        text_probs: Dict[str, float],
        fonts: List[str],
        backgrounds: List[str],
    ):
        self.num_samples = num_samples
        self.bag_of_words = bag_of_words
        self.text_probs = text_probs    
        self.fonts = fonts
        self.backgrounds = backgrounds

        self.punctuations = ['-', '<', '/', ',', "'", ':', '&', '.', '(', ')']
    
    
    def generate(
        self,
        font_path: str = None,
        font_size: int = 22,
        font_weights: List[float] = None,
        text: str = None,
        max_num_words: int = Config.max_num_words,
        two_lined_samples: bool = False,
        get_text_only: bool = False,
    ) -> Tuple[str, Image]:

        isText = True
        toCase = str.upper
        if font_path is None:
            font, font_path = get_pil_font(self.fonts, font_size=font_size, font_weights=font_weights)
        else:
            font = ImageFont.truetype(font_path, font_size)
        
        if text is None:
            toCase = None
            isText = False
            toGenerate: str = random.choices(list(self.text_probs.keys()), weights=self.text_probs.values(), k=1)[0]
            if toGenerate == "text":
                # generate text
                text = (random.randint(1, 2) * ' ').join(random.sample(
                    self.bag_of_words, 
                    random.choice([*list(range(1, max_num_words)), *[1, 1]])
                ))
                if two_lined_samples:
                    text = getTwoLined(text)     # get two lines text, can also return single line
                    
                toCase = random.choices(
                    population=[str.upper, str.lower, str.capitalize, str.title], weights=[0.8, 0.05, 0.05, 0.1], k=1
                )[0]
                text = toCase(text)
                # add random punctuations
                if random.uniform(0, 1) > 0.7:
                    punct = random.choice(self.punctuations)
                    if punct not in ['<']:                
                        si = random.choice([*[i for i, x in enumerate(text) if x == ' '], *[0, len(text) - 1]])
                        ps = random.choice([f' {punct}', f' {punct} ', f'{punct} '])
                        text = text[:si] + ps + text[si + 1:]
                        
                if random.uniform(0, 1) > 0.5 and toCase == str.upper:
                    text = text.split()[0] + ", " + " ".join(text.split()[1:])

                # add specified chars to the words randomly
                if random.uniform(0, 1) > 0.7 and toCase == str.upper:
                    for _ in range(3):
                        char = random.choice(['I', 'Q', 'O', '0', 'W', 'M', 'N', 'V', 'X', 'B', 'L', 'T', 'AA', '3', '8'])
                        si = random.choice([*[i for i, x in enumerate(text) if x == ' '], *[0, len(text) - 1]])
                        ps = random.choice([f' {char}', f'{char} '])
                        text = text[:si] + ps + text[si + 1:]
                isText = True
            elif toGenerate == "date":
                text = generate_random_date()
            else:
                # generate number
                l = []
                for _ in range(random.randint(1, 15)):
                    if random.random() > 0.3:
                        letter = string.digits[random.randint(0, 9)]
                    else:
                        letter = random.choice([string.ascii_uppercase[random.randint(0, 25)], "-"])
                    l.append(letter)
                text = ''.join(l)
        
        if get_text_only:
            return ' '.join(text.split()), None
        
        # textsize = font.getsize(text.split("\n")[0])
        char_freq = Counter(text)
        letter_spacing = 0
        # add random spacing
        if isText and toCase == str.upper:
            if random.random() > 0.4:
                letter_spacing = random.randint(1, 5)
        
        # get image size
        img_w, img_h = 0, 0
        for char in text.split("\n")[0]:
            bbox = font.getbbox(char)  # returns (x0, y0, x1, y1)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            # char_width, char_height = font.getsize(char)
            img_w += char_width + letter_spacing
            img_h = max(char_height, img_h)
        
        offset_x, offset_y = int(random.uniform(0.01, 0.1)*img_w), int(random.uniform(0.2, 0.5) * img_h)
        size = (img_w + offset_x, img_h * (char_freq['\n'] + 1) + offset_y)
        
        img = add_background(size, self.backgrounds) if random.uniform(0, 1) > 0.2 else Image.new('L', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Initial x and y positions
        x, y = random.randint(0, offset_x) , random.randint(0, offset_y//3)
        init_x = x
        # Draw each character with custom horizontal spacing and, slight location change in y
        ofst_y__ = (img_h - char_height) / len(text)
        ofst_y__ = (random.choice([1, -1]) *  ofst_y__)
        color = random.choice(["#2f2f2f", "black", "#404040"])
        for char in text:
            if char == "\n":
                y += char_height
                x = init_x
                continue
            # char_width, char_height = draw.textsize(char, font=font)
            bbox = draw.textbbox((0, 0), char, font=font)  # ✅ new method
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            draw.text((x, y), char, fill=color, font=font, align="center")
            x += char_width + letter_spacing
            y += ofst_y__
        img = img.convert("RGB")
        
        # apply blur
        if random.random() > 0.3:
            _, font_name = os.path.split(font_path)
            img = img.filter(ImageFilter.GaussianBlur(0.7))
        
        isUpper = toCase == str.upper
        if random.random() > 0.6 and isUpper:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
        
        # apply blur
        if random.random() > 0.6:
            for op in random.choices([guassianBlur, motionBlur, bokenBlur], k=2):
                img = op(img, img_w)
        elif random.random() > 0.4:    # apply motion blur  
            img = motionBlur(img)
        elif random.random() > 0.2:    # boken blur
            img = bokenBlur(img, img_w)
        else:
            img = guassianBlur(img)
            
        img = img.convert("RGB")
        # resize image randomly
        new_size = None
        if random.random() > 0.7:
            resize_factor = random.uniform(0.9, 1)
            w, h = img.size
            new_size = (int(w * resize_factor), int(h * resize_factor))
        
        # stretching both sides
        if random.random() > 0.6:
            w, h = new_size if new_size is not None else img.size
            if random.random() > 0.5:
                new_h = h + int(h * random.uniform(0.1, 0.3))
            else:
                new_h = h - int(h * random.uniform(0.05, 0.2))   
            img = img.resize((w, new_h))
        else:
            if new_size is not None:
                img = img.resize(new_size)
        
        # crop randomly
        if random.random() > 0.5:
            w, h = img.size
            x1 = random.uniform(0.008, 0.01) * w
            y1 = random.uniform(0.01, 0.1) * h
            x2 = w - (random.uniform(0.008, 0.01) * w)
            y2 = h - (random.uniform(0.01, 0.1) * h)
            img = img.crop((x1, y1, x2, y2))
        
        # add some opacity
        if random.random() > 0.5:
            img.putalpha(random.randint(150, 210))

        if random.random() > 0.7:
            img = add_moire_patterns(img, alpha=random.uniform(0.1, 0.3))
        
        # change brighness randomly
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.2))
        
        # remove the extra spaces if any
        text = ' '.join(text.split())
        return text, img.convert("RGB")