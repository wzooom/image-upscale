# Real-ESRGAN Image Upscaler API

This project provides a FastAPI-based web API for upscaling images using the Real-ESRGAN model.

## Features
- Upload an image and receive an upscaled version using Real-ESRGAN (x4 scale).
- Runs entirely on CPU by default.

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

### 4. Download Model Weights (Optional)
The first time you run the API, it will attempt to download the Real-ESRGAN model weights automatically if they are not present (`RealESRGAN_x4.pth`).

### 5. Run the API Server
Start the FastAPI server using Uvicorn:
```bash
python upscale_api.py
```
By default, the server will be available at [http://localhost:8000](http://localhost:8000).

### 6. Upscale an Image
Send a POST request to `/upscale/` with an image file. Example using `curl`:
```bash
curl -X POST "http://localhost:8000/upscale/" -F "file=@your_image.png" --output upscaled.png
```

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'realesrgan'`, make sure your virtual environment is activated and requirements are installed.
- If you have multiple Python installations, ensure you are using the correct `python` and `pip` from your virtual environment.

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies

## License
MIT License
