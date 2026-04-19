#!/usr/bin/env python3
"""
End-to-end test driver for upscale_api_unified.

Starts the server in a subprocess, runs assertions against /status and /upscale,
tears the server down, exits non-zero on any failure, prints PASS N/N on success.
"""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

import image_utils

HOST = "127.0.0.1"
PORT = 8765
BASE_URL = f"http://{HOST}:{PORT}"
TEST_IMAGE_SIZE = (100, 100)

_passes = 0
_fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passes
    if cond:
        _passes += 1
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f"  — {detail}" if detail else "")
        _fails.append(msg)
        print(msg)


def make_test_image() -> Path:
    image_utils.ensure_dirs_exist()
    path = Path(image_utils.get_input_path("test_image.png"))
    img = Image.new("RGB", TEST_IMAGE_SIZE, color="blue")
    draw = ImageDraw.Draw(img)
    w, h = TEST_IMAGE_SIZE
    draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill="red")
    draw.line([(0, 0), (w - 1, h - 1)], fill="white", width=2)
    img.save(path)
    return path


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import uvicorn; from upscale_api_unified import app; "
         f"uvicorn.run(app, host='{HOST}', port={PORT}, log_level='warning')"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise RuntimeError(f"Server exited before becoming ready:\n{out}")
        try:
            r = httpx.get(f"{BASE_URL}/status", timeout=1.0)
            if r.status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    proc.terminate()
    raise RuntimeError("Server failed to become ready within 30s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def post_image(image_path: Path, params: dict) -> httpx.Response:
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f.read(), "image/png")}
    return httpx.post(f"{BASE_URL}/upscale/", params=params, files=files, timeout=120.0)


def assert_png_size(resp_content: bytes, expected: tuple[int, int]) -> tuple[bool, str]:
    try:
        img = Image.open(io.BytesIO(resp_content))
        img.load()
    except Exception as e:
        return False, f"not a valid image: {e}"
    if img.size != expected:
        return False, f"size {img.size} != expected {expected}"
    return True, ""


def run(image_path: Path) -> None:
    # /status
    r = httpx.get(f"{BASE_URL}/status", timeout=5.0)
    check("GET /status -> 200", r.status_code == 200, str(r.status_code))
    status = r.json()
    check("status payload has realesrgan_available",
          isinstance(status.get("realesrgan_available"), bool))
    pillow_methods = status.get("pillow_methods", [])
    check("status lists 4 pillow methods", len(pillow_methods) == 4,
          f"got {pillow_methods}")
    realesrgan_available = bool(status.get("realesrgan_available"))

    # /
    r = httpx.get(f"{BASE_URL}/", timeout=5.0)
    check("GET / -> 200", r.status_code == 200)

    # Pillow methods × {2, 4}
    out_simple = image_utils.OUTPUT_DIR / "simple" if hasattr(image_utils, "OUTPUT_DIR") else Path("images/output/simple")
    out_simple.mkdir(parents=True, exist_ok=True)
    for method in pillow_methods:
        for scale in (2, 4):
            name = f"POST /upscale method={method} scale={scale}"
            r = post_image(image_path, {"method": method, "scale": scale})
            if r.status_code != 200:
                check(name, False, f"status={r.status_code} body={r.text[:200]}")
                continue
            ok, detail = assert_png_size(
                r.content, (TEST_IMAGE_SIZE[0] * scale, TEST_IMAGE_SIZE[1] * scale)
            )
            check(name, ok, detail)
            if ok:
                (out_simple / f"{method}_x{scale}.png").write_bytes(r.content)

    # Real-ESRGAN
    out_re = Path("images/output/realesrgan")
    out_re.mkdir(parents=True, exist_ok=True)
    if realesrgan_available:
        for model in status.get("realesrgan_models", []):
            name = f"POST /upscale method=realesrgan model={model}"
            r = post_image(image_path, {"method": "realesrgan", "model": model})
            if r.status_code != 200:
                check(name, False, f"status={r.status_code} body={r.text[:200]}")
                continue
            ok, detail = assert_png_size(
                r.content, (TEST_IMAGE_SIZE[0] * 4, TEST_IMAGE_SIZE[1] * 4)
            )
            check(name, ok, detail)
            if ok:
                (out_re / f"{model}.png").write_bytes(r.content)

        # invalid model
        r = post_image(image_path, {"method": "realesrgan", "model": "nope_does_not_exist"})
        check("invalid realesrgan model -> 400", r.status_code == 400,
              f"status={r.status_code}")
    else:
        print("  SKIP  Real-ESRGAN not installed in this env")
        r = post_image(image_path, {"method": "realesrgan"})
        check("realesrgan requested when unavailable -> 503",
              r.status_code == 503, f"status={r.status_code} body={r.text[:200]}")

    # Error cases
    r = post_image(image_path, {"method": "garbage"})
    check("invalid method -> 400", r.status_code == 400, f"status={r.status_code}")

    r = post_image(image_path, {"method": "lanczos", "scale": 99})
    check("invalid scale -> 400", r.status_code == 400, f"status={r.status_code}")

    # Non-image content type
    r = httpx.post(
        f"{BASE_URL}/upscale/",
        files={"file": ("x.txt", b"not an image", "text/plain")},
        timeout=10.0,
    )
    check("non-image content_type -> 400", r.status_code == 400, f"status={r.status_code}")


def main() -> int:
    print("=== Unified Upscaling API test harness ===")
    image_path = make_test_image()
    print(f"test image: {image_path}")
    proc = start_server()
    try:
        run(image_path)
    finally:
        stop_server(proc)

    total = _passes + len(_fails)
    print()
    if _fails:
        print(f"FAILED: {len(_fails)}/{total}")
        for line in _fails:
            print(line)
        return 1
    print(f"PASS {_passes}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
