"""
Utility module for managing image file paths in the project.
Provides functions to get standardized paths for input, output, and temporary images.
"""

import os
import shutil
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = BASE_DIR / "images"
INPUT_DIR = IMAGES_DIR / "input"
OUTPUT_DIR = IMAGES_DIR / "output"
TEMP_DIR = IMAGES_DIR / "temp"

# Output subdirectories
REALESRGAN_OUTPUT_DIR = OUTPUT_DIR / "realesrgan"
SIMPLE_OUTPUT_DIR = OUTPUT_DIR / "simple"

# Ensure directories exist
def ensure_dirs_exist():
    """Ensure all required directories exist."""
    for directory in [INPUT_DIR, REALESRGAN_OUTPUT_DIR, SIMPLE_OUTPUT_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Path getters
def get_input_path(filename):
    """Get the full path for an input image."""
    return INPUT_DIR / filename

def get_output_path(api_type, filename):
    """
    Get the full path for an output image.
    
    Args:
        api_type: Either 'realesrgan' or 'simple'
        filename: The output filename
    """
    if api_type.lower() == 'realesrgan':
        return REALESRGAN_OUTPUT_DIR / filename
    else:
        return SIMPLE_OUTPUT_DIR / filename

def get_temp_path(filename):
    """Get the full path for a temporary image."""
    return TEMP_DIR / filename

def move_existing_test_images():
    """Move existing test images to the new directory structure."""
    # Move test_image.png to input directory if it exists
    if os.path.exists(BASE_DIR / "test_image.png"):
        shutil.copy2(BASE_DIR / "test_image.png", INPUT_DIR / "test_image.png")
        print(f"Moved test_image.png to {INPUT_DIR}")
    
    # Move upscaled images to the simple output directory
    for filename in os.listdir(BASE_DIR):
        if filename.startswith("upscaled_") and filename.endswith(".png"):
            shutil.copy2(BASE_DIR / filename, SIMPLE_OUTPUT_DIR / filename)
            print(f"Moved {filename} to {SIMPLE_OUTPUT_DIR}")
    
    # Move test results to the appropriate output directories
    if os.path.exists(BASE_DIR / "test_results"):
        for filename in os.listdir(BASE_DIR / "test_results"):
            if filename.startswith("realesrgan_"):
                shutil.copy2(BASE_DIR / "test_results" / filename, REALESRGAN_OUTPUT_DIR / filename)
                print(f"Moved test_results/{filename} to {REALESRGAN_OUTPUT_DIR}")

def clean_temp_directory():
    """Clean the temporary directory."""
    for file in TEMP_DIR.glob("*"):
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")

# Initialize directories when module is imported
ensure_dirs_exist()
