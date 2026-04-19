#!/usr/bin/env python3
"""
Unified Image Upscaling API
---------------------------
FastAPI service that upscales images using either Real-ESRGAN (AI, slow,
high quality) or Pillow resampling (fast, always available).

The service auto-detects whether Real-ESRGAN is installed and falls back
to Pillow methods if it isn't.
"""

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError, ImageFilter
import io
import os
import importlib
import logging
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger("upscale_api_unified")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Unified Image Upscaling API",
    description="API for upscaling images using Real-ESRGAN or Pillow-based methods",
    version="1.1.0",
)


class UpscalingMethod(str, Enum):
    REALESRGAN = "realesrgan"
    LANCZOS = "lanczos"
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"


PILLOW_METHODS = {
    UpscalingMethod.LANCZOS.value: Image.Resampling.LANCZOS,
    UpscalingMethod.BICUBIC.value: Image.Resampling.BICUBIC,
    UpscalingMethod.BILINEAR.value: Image.Resampling.BILINEAR,
    UpscalingMethod.NEAREST.value: Image.Resampling.NEAREST,
}

REALESRGAN_MODELS = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file_name": "RealESRGAN_x4plus.pth",
        "num_block": 23,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file_name": "RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
    },
}


def _check_realesrgan_available() -> bool:
    for module in ("realesrgan", "basicsr", "torch", "cv2"):
        try:
            importlib.import_module(module)
        except ImportError:
            return False
    return True


REALESRGAN_AVAILABLE = _check_realesrgan_available()

# Lazy-initialized, keyed by model name.
_upsampler_cache: dict = {}


def initialize_upsampler(model_name: str = "RealESRGAN_x4plus"):
    """Download weights if needed and build a RealESRGANer. Cached per model."""
    if not REALESRGAN_AVAILABLE:
        raise RuntimeError("Real-ESRGAN is not available in this environment")
    if model_name not in REALESRGAN_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported: {list(REALESRGAN_MODELS.keys())}"
        )
    if model_name in _upsampler_cache:
        return _upsampler_cache[model_name]

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url

    config = REALESRGAN_MODELS[model_name]
    model_path = config["file_name"]
    if not os.path.isfile(model_path):
        model_path = load_file_from_url(
            url=config["url"], model_dir=".", progress=True, file_name=config["file_name"]
        )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=config["num_block"], num_grow_ch=32, scale=4,
    )
    upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model,
        tile=0, tile_pad=10, pre_pad=0, half=False,
    )
    _upsampler_cache[model_name] = upsampler
    return upsampler


@app.get("/")
async def root():
    return {
        "message": "Unified Image Upscaling API",
        "methods_available": {
            "realesrgan": REALESRGAN_AVAILABLE,
            "pillow": True,
        },
        "usage": "POST to /upscale/ with an image file",
    }


@app.get("/status")
async def status():
    return {
        "status": "running",
        "realesrgan_available": REALESRGAN_AVAILABLE,
        "pillow_methods": list(PILLOW_METHODS.keys()),
        "realesrgan_models": list(REALESRGAN_MODELS.keys()) if REALESRGAN_AVAILABLE else [],
    }


def upscale_with_realesrgan(image: Image.Image, model: str = "RealESRGAN_x4plus") -> Image.Image:
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Real-ESRGAN is not available in this environment")

    import cv2  # safe: guarded by REALESRGAN_AVAILABLE

    try:
        upsampler = initialize_upsampler(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Real-ESRGAN: {e}")

    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out_bgr, _ = upsampler.enhance(img_bgr, outscale=4)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def upscale_with_pillow(image: Image.Image, method: str = "lanczos", scale: int = 4) -> Image.Image:
    if method not in PILLOW_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid upscaling method. Available methods: {', '.join(PILLOW_METHODS)}",
        )
    width, height = image.size
    sharpened = image.filter(ImageFilter.SHARPEN)
    return sharpened.resize((width * scale, height * scale), resample=PILLOW_METHODS[method])


@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    method: Optional[str] = Query(None, description="Upscaling method (realesrgan, lanczos, bicubic, bilinear, nearest)"),
    scale: int = Query(4, description="Scale factor for Pillow methods (1-8)"),
    model: str = Query("RealESRGAN_x4plus", description="Real-ESRGAN model to use"),
):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    if scale < 1 or scale > 8:
        raise HTTPException(status_code=400, detail="Scale factor must be between 1 and 8.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file. Please upload a valid image.")

    selected = method
    if selected is None:
        selected = UpscalingMethod.REALESRGAN.value if REALESRGAN_AVAILABLE else UpscalingMethod.LANCZOS.value

    try:
        if selected == UpscalingMethod.REALESRGAN.value:
            result = upscale_with_realesrgan(image, model)
        elif selected in PILLOW_METHODS:
            result = upscale_with_pillow(image, selected, scale)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid upscaling method. Available: realesrgan, {', '.join(PILLOW_METHODS)}",
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upscale image: {e}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting Unified Image Upscaling API (Real-ESRGAN %s)",
        "available" if REALESRGAN_AVAILABLE else "NOT available — Pillow only",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
