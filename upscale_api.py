from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from PIL import Image, UnidentifiedImageError
import io
import os
import numpy as np
import cv2

# Real-ESRGAN imports
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

app = FastAPI()

# Model configurations
MODEL_CONFIGS = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file_name": "RealESRGAN_x4plus.pth",
        "num_block": 23
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file_name": "RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6
    }
}

# Initialize Real-ESRGAN model with automatic downloading
def initialize_upsampler(model_name: str = "RealESRGAN_x4plus"):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["RealESRGAN_x4plus"])
    model_path = config["file_name"]
    
    # Download model if not present
    if not os.path.isfile(model_path):
        try:
            model_path = load_file_from_url(
                url=config["url"], 
                model_dir=".", 
                progress=True, 
                file_name=config["file_name"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download model {model_name}: {str(e)}")
    
    # Initialize model based on type
    if model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=config["num_block"], num_grow_ch=32, scale=4)
    else:  # Default to RealESRGAN_x4plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=config["num_block"], num_grow_ch=32, scale=4)
    
    try:
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        return upsampler
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Real-ESRGAN model: {str(e)}")

# Initialize the default upsampler (this will download the model if needed)
upsampler = initialize_upsampler()

@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    model: str = Query("RealESRGAN_x4plus", description="Model to use for upscaling")
):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")
    
    try:
        contents = await file.read()
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL Image to numpy array (RGB to BGR for OpenCV)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize upsampler for the selected model if different from default
        current_upsampler = upsampler
        if model != "RealESRGAN_x4plus" and model in MODEL_CONFIGS:
            current_upsampler = initialize_upsampler(model)
        
        # Upscale the image using Real-ESRGAN
        upscaled_img, _ = current_upsampler.enhance(img, outscale=4)
        
        # Convert BGR back to RGB
        upscaled_img = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array back to PIL Image
        result_image = Image.fromarray(upscaled_img)
        
        # Save upscaled image to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file. Please upload a valid image.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upscale image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("upscale_api:app", host="0.0.0.0", port=8000, reload=True)
