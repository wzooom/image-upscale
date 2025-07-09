from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import os

# Import Real-ESRGAN
from realesrgan import RealESRGAN


app = FastAPI()

# Initialize the Real-ESRGAN model (CPU)
def get_model():
    model = RealESRGAN('cpu', scale=4)
    model.load_weights('RealESRGAN_x4.pth', download=True)  # Download weights if not present
    return model

@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Load model and enhance image
    model = get_model()
    upscaled_image = model.predict(input_image)
    
    # Prepare output
    buf = io.BytesIO()
    upscaled_image.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("upscale_api:app", host="0.0.0.0", port=8000, reload=True)
