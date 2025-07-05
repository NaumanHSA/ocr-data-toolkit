"""
Example usage script for OCR Data Toolkit

This script demonstrates how to:
- Generate a single synthetic OCR image and save it
- Generate a full training/testing dataset
- Visualize the font catalog

Adjust the configuration as needed for your use case.
"""

import os
from ocr_data_toolkit import ODT


if __name__ == "__main__":
    # --- 1. Minimal Example: Generate and save a single image ---
    print("Generating a single synthetic OCR image...")
    odt = ODT(language="en")
    text, image = odt.generate_single_image()
    image.save("sample_ocr.png")
    print(f"Saved: sample_ocr.png | Ground Truth: {text}")

    # --- 2. Generate a full dataset (train/test split) ---
    print("\nGenerating a dataset of 100 samples (with train/test split)...")
    dataset_odt = ODT(
        language="en",
        output_image_size=(128, 32),
        num_workers=2,
        augmentation_config={
            "max_num_words": 6,
            "num_lines": 2,
            "font_size": 36,
            "blur_probs": {"gaussian": 0.3, "custom_blurs": 0.5},
        }
    )
    dataset_odt.generate_training_data(num_samples=100)
    print("Dataset generated in:", dataset_odt.output_save_path)
    print("  - Train images:", os.path.join(dataset_odt.train_path, "images"))
    print("  - Test images:", os.path.join(dataset_odt.test_path, "images"))
    print("  - Train GT:", os.path.join(dataset_odt.train_path, "gt.txt"))
    print("  - Test GT:", os.path.join(dataset_odt.test_path, "gt.txt"))

    # --- 3. Visualize font catalog ---
    print("\nVisualizing available fonts...")
    catalog_dir = "font_catalog"
    odt.visualize_font_catalog(save_dir=catalog_dir, chunk_size=10)
    print(f"Font catalog visualizations saved in: {os.path.abspath(catalog_dir)}")
    print("\nDone.")