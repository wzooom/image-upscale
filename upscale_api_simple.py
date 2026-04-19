from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from PIL import Image, UnidentifiedImageError, ImageFilter
import io
import os
import numpy as np

app = FastAPI()

# Available upscaling methods
UPSCALE_METHODS = {
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST
}

@app.get("/")
async def root():
    return {"message": "Image Upscaling API is running. Use /upscale/ endpoint to upscale images."}

@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Query(4, description="Scale factor for upscaling (1-8)"),
    method: str = Query("lanczos", description="Upscaling method to use")
):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")
    
    # Validate scale factor
    if scale < 1 or scale > 8:
        raise HTTPException(status_code=400, detail="Scale factor must be between 1 and 8.")
    
    # Validate upscaling method
    if method not in UPSCALE_METHODS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid upscaling method. Available methods: {', '.join(UPSCALE_METHODS.keys())}"
        )
    
    try:
        contents = await file.read()
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Get original dimensions
        width, height = image.size
        
        # Calculate new dimensions
        new_width = width * scale
        new_height = height * scale
        
        # Apply slight sharpening to improve upscaled image quality
        image = image.filter(ImageFilter.SHARPEN)
        
        # Upscale the image using the selected method
        upscaled_img = image.resize((new_width, new_height), resample=UPSCALE_METHODS[method])
        
        # Save upscaled image to bytes
        img_byte_arr = io.BytesIO()
        upscaled_img.save(img_byte_arr, format='PNG')
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
    uvicorn.run("upscale_api_simple:app", host="0.0.0.0", port=8000, reload=True)
