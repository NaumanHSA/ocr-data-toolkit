import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Canvas
w, h = 1200, 350
img = Image.new('RGB', (w, h), color=(18, 18, 23))
draw = ImageDraw.Draw(img)

# Title centered at the top
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
title = 'OCR Data Toolkit'
font_title = ImageFont.truetype(font_path, 82)
# Center title using textbbox for accurate sizing
bbox_title = draw.textbbox((0, 0), title, font=font_title)
title_w = bbox_title[2] - bbox_title[0]
title_h = bbox_title[3] - bbox_title[1]
draw.text(((w-title_w)//2, 28), title, font=font_title, fill=(255, 208, 60))

# Features (top 5)
features = [
    '• Synthetic OCR Images',
    '• Realistic Augmentations',
    '• Multiprocessing Support',
    '• Custom Fonts & Backgrounds',
    '• Train/Test Dataset Generation',
]
font_feat = ImageFont.truetype(font_path, 22)
feat_y = 60 + title_h + 20
for i, feat in enumerate(features):
    bbox_feat = draw.textbbox((0, 0), feat, font=font_feat)
    feat_w = bbox_feat[2] - bbox_feat[0]
    feat_h = bbox_feat[3] - bbox_feat[1]
    draw.text((w//3, feat_y + i*38), feat, font=font_feat, fill=(180, 220, 255))

# Save
os.makedirs('docs', exist_ok=True)
img.save('docs/cover.png')
print('Cover image generated at docs/cover.png')
