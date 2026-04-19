# Unified Image Upscaler API

This project provides a streamlined FastAPI-based web API for upscaling images using either Real-ESRGAN models or standard PIL-based upscaling methods. The unified architecture automatically selects the best available method based on your environment and preferences.

## Features

### File System Structure
The project uses an organized file system structure for managing test images:

```
images/
├── input/           # Original test images
├── output/          # Upscaled images
│   ├── realesrgan/  # Results from Real-ESRGAN API
│   └── simple/      # Results from simplified API
└── temp/            # Temporary images
```

The `image_utils.py` module provides utility functions for working with this file system:
- `get_input_path(filename)`: Get the path for an input image
- `get_output_path(api_type, filename)`: Get the path for an output image (api_type can be 'realesrgan' or 'simple')
- `get_temp_path(filename)`: Get the path for a temporary image

### Unified API (`upscale_api_unified.py`)
- Single API that supports both Real-ESRGAN and Pillow-based upscaling methods
- Automatically detects available dependencies and capabilities
- Intelligent fallback to Pillow methods when Real-ESRGAN is unavailable

#### Real-ESRGAN Features
- Upload an image and receive an upscaled version using Real-ESRGAN (x4 scale)
- Runs entirely on CPU by default
- Supports multiple Real-ESRGAN models:
  - `RealESRGAN_x4plus`: General purpose model for most images
  - `RealESRGAN_x4plus_anime_6B`: Optimized for anime/illustration images

#### Pillow-based Features
- Uses standard PIL (Pillow) image processing for upscaling
- Supports multiple upscaling methods:
  - `lanczos`: High-quality upscaling (default)
  - `bicubic`: Good quality with smoother results
  - `bilinear`: Medium quality, faster processing
  - `nearest`: Lower quality, fastest processing
- Configurable scale factor (1-8x)

## Setup Instructions

### 1. Clone the Repository
If you haven't already, clone this repository and navigate to the project directory:
```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2. Create and Activate a Virtual Environment
It is recommended to use a Python virtual environment to manage dependencies:
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Requirements
Install all required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Model Weights (Real-ESRGAN version only)
The Real-ESRGAN implementation will automatically download model weights on first run if they are not present in the project directory. This step is not required for the simplified PIL-based implementation.

### 5. File System Structure
The project uses an organized file system structure for test images. When you run any of the test scripts, the necessary directories will be automatically created:

- `images/input/`: Contains original test images
- `images/output/realesrgan/`: Contains upscaled images from the Real-ESRGAN API
- `images/output/simple/`: Contains upscaled images from the simplified API
- `images/temp/`: Contains temporary images used during processing

The `image_utils.py` module provides utility functions for working with this file system.

### 6. Run the API Server
Start the unified FastAPI server using Uvicorn:

```bash
python upscale_api_unified.py
```

The server will automatically detect available dependencies and capabilities. By default, the server will be available at [http://localhost:8000](http://localhost:8000).

You can check the status and available methods by accessing the `/status` endpoint.

### 6a. Run the end-to-end test harness
With the dependencies installed, you can exercise every method and error path in one shot:

```bash
python test_unified_api.py
```

The harness spawns the server on port 8765, hits `/status` and `/upscale/` with every valid combination (plus error cases), writes sample outputs under `images/output/`, tears the server down, and prints `PASS N/N` on success. It exits non-zero if anything fails.

### 7. Upscale an Image
Send a POST request to `/upscale/` with an image file.

```bash
# Using the default method (Real-ESRGAN if available, otherwise Lanczos)
curl -X POST "http://localhost:8000/upscale/" -F "file=@your_image.png" --output upscaled.png

# Explicitly using Real-ESRGAN with a specific model
curl -X POST "http://localhost:8000/upscale/?method=realesrgan&model=RealESRGAN_x4plus_anime_6B" -F "file=@your_image.png" --output upscaled.png

# Using Pillow-based method with specific parameters
curl -X POST "http://localhost:8000/upscale/?method=bicubic&scale=2" -F "file=@your_image.png" --output upscaled.png
```

## Troubleshooting

### General Issues
- If you have multiple Python installations, ensure you are using the correct `python` and `pip` from your virtual environment.
- Check the API status by accessing the `/status` endpoint to see which methods are available.

### Real-ESRGAN Issues
- If Real-ESRGAN is not available, the API will automatically fall back to Pillow methods and `/status` reports `realesrgan_available: false`. Requesting `method=realesrgan` against such a server returns HTTP 503.
- Real-ESRGAN is verified working on Python 3.11 with `torch==2.2.x`, `torchvision==0.17.x`, `realesrgan==0.3.0`, `basicsr==1.4.2`.
- `basicsr` 1.4.2 ships an import of `torchvision.transforms.functional_tensor`, which was removed in `torchvision>=0.17`. If you see `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` at import time, patch the single line in your installed `basicsr/data/degradations.py` to import from `torchvision.transforms.functional` instead.
- If you encounter memory issues when upscaling large images, try raising the `tile` parameter in the `RealESRGANer` initialization inside `upscale_api_unified.py`.
- Python 3.13+ is not supported by the current `realesrgan` + `basicsr` combo. Use Python 3.10–3.12.

### Pillow-based Methods
- For best quality results, use the `lanczos` method with a scale factor of 2-4.
- For faster processing with acceptable quality, use the `bicubic` method.

## Requirements

### Core Dependencies (Required)
- Python 3.8+
- Dependencies:
  - fastapi
  - uvicorn
  - Pillow
  - numpy
  - requests

### Real-ESRGAN Dependencies (Optional)
- Python 3.8-3.12 recommended (compatibility issues may occur with Python 3.13+)
- Additional dependencies:
  - realesrgan
  - torch
  - torchvision
  - opencv-python

The unified API will automatically detect which dependencies are available and use the appropriate upscaling methods.

## License
MIT License
