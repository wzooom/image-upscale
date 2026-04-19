#!/usr/bin/env python3
"""
Unified test script for both upscaling API implementations:
1. Real-ESRGAN implementation (upscale_api.py)
2. Simplified Pillow implementation (upscale_api_simple.py)

This script will:
1. Create a test image
2. Test both API implementations if available
3. Generate a comparison report
"""

import os
import sys
import time
import subprocess
import requests
import signal
from PIL import Image, ImageDraw
import io
import importlib.util
import image_utils

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE_PATH = str(image_utils.get_input_path("test_image.png"))

def create_test_dir():
    """Create output directories if they don't exist"""
    image_utils.ensure_dirs_exist()
    print("Ensured all image directories exist")

def create_test_image(size=(100, 100)):
    """Create a simple test image with a blue background and red square"""
    img = Image.new('RGB', size, color='blue')
    draw = ImageDraw.Draw(img)
    
    # Draw a red square in the center (25% to 75% of the image)
    x1 = size[0] // 4
    y1 = size[1] // 4
    x2 = size[0] * 3 // 4
    y2 = size[1] * 3 // 4
    draw.rectangle([x1, y1, x2, y2], fill='red')
    
    # Save the image to the input directory
    output_path = image_utils.get_input_path("test_image.png")
    img.save(output_path)
    print(f"Test image created: {output_path}")
    return str(output_path)

def check_module_availability(module_name):
    """Check if a Python module can be imported"""
    try:
        importlib.util.find_spec(module_name)
        return True
    except ImportError:
        return False

def start_api_server(api_script):
    """Start the API server as a subprocess"""
    print(f"Starting {api_script} server...")
    process = subprocess.Popen(
        [sys.executable, api_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}")
        if response.status_code == 200:
            print(f"Server started successfully: {api_script}")
            return process
        else:
            print(f"Server returned unexpected status code: {response.status_code}")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return None
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to server: {api_script}")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return None

def stop_api_server(process):
    """Stop the API server subprocess"""
    if process:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        print("Server stopped")

def test_upscale_api(api_type, params=None):
    """Test the upscale API with the given parameters"""
    if params is None:
        params = {}
    
    url = f"{API_URL}/upscale/"
    if params:
        url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': (os.path.basename(TEST_IMAGE_PATH), f, 'image/png')}
        
        try:
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                # Generate output filename based on parameters
                param_str = "_".join([f"{k}_{v}" for k, v in params.items()]) if params else "default"
                output_filename = f"{api_type}_{param_str}.png"
                output_path = image_utils.get_output_path(api_type, output_filename)
                
                with open(output_path, 'wb') as out_file:
                    out_file.write(response.content)
                print(f"Image upscaled successfully! Saved as {output_path}")
                return True, str(output_path)
            else:
                print(f"Error: {response.status_code}")
                try:
                    print(response.json())
                except:
                    print("Could not parse error response as JSON")
                return False, None
        except requests.exceptions.ConnectionError:
            print("Failed to connect to server")
            return False, None

def test_realesrgan_api():
    """Test the Real-ESRGAN API implementation"""
    print("\n=== Testing Real-ESRGAN API (upscale_api.py) ===")
    
    # Check if required modules are available
    required_modules = ['realesrgan', 'basicsr', 'torch', 'cv2']
    missing_modules = [m for m in required_modules if not check_module_availability(m)]
    
    if missing_modules:
        print(f"Cannot test Real-ESRGAN API: Missing required modules: {', '.join(missing_modules)}")
        print("Please install these modules or use the simplified API instead.")
        return False
    
    # Start the server
    process = start_api_server('upscale_api.py')
    if not process:
        return False
    
    try:
        # Test default model
        test_upscale_api('realesrgan', {})
        
        # Test anime model if available
        test_upscale_api('realesrgan', {'model': 'RealESRGAN_x4plus_anime_6B'})
        
        # Test error handling
        test_upscale_api('realesrgan', {'model': 'invalid_model'})
        
        return True
    finally:
        stop_api_server(process)

def test_simple_api():
    """Test the simplified Pillow-based API implementation"""
    print("\n=== Testing Simplified API (upscale_api_simple.py) ===")
    
    # Start the server
    process = start_api_server('upscale_api_simple.py')
    if not process:
        return False
    
    try:
        # Test default method and scale
        test_upscale_api('simple', {})
        
        # Test different methods
        for method in ['lanczos', 'bicubic', 'bilinear', 'nearest']:
            test_upscale_api('simple', {'method': method})
        
        # Test different scales
        for scale in [2, 4, 8]:
            test_upscale_api('simple', {'scale': scale})
        
        # Test combinations
        test_upscale_api('simple', {'method': 'bicubic', 'scale': 2})
        
        # Test error handling
        test_upscale_api('simple', {'method': 'invalid_method'})
        test_upscale_api('simple', {'scale': 10})
        
        return True
    finally:
        stop_api_server(process)

def compare_results():
    """Compare the results of both API implementations"""
    print("\n=== Comparing Results ===")
    
    # Check if we have results from both implementations
    realesrgan_dir = image_utils.REALESRGAN_OUTPUT_DIR
    simple_dir = image_utils.SIMPLE_OUTPUT_DIR
    
    realesrgan_results = [f for f in os.listdir(realesrgan_dir) if f.startswith('realesrgan_')]
    simple_results = [f for f in os.listdir(simple_dir) if f.startswith('simple_')]
    
    if not realesrgan_results:
        print("No Real-ESRGAN results found for comparison")
    
    if not simple_results:
        print("No Simplified API results found for comparison")
    
    if not realesrgan_results or not simple_results:
        return
    
    # Compare default results if available
    realesrgan_default = next((f for f in realesrgan_results if 'default' in f), None)
    simple_default = next((f for f in simple_results if 'default' in f), None)
    
    if realesrgan_default and simple_default:
        print("\nComparison of default upscaling:")
        realesrgan_img = Image.open(os.path.join(realesrgan_dir, realesrgan_default))
        simple_img = Image.open(os.path.join(simple_dir, simple_default))
        
        print(f"Real-ESRGAN size: {realesrgan_img.size}")
        print(f"Simplified API size: {simple_img.size}")
        print(f"Real-ESRGAN format: {realesrgan_img.format}")
        print(f"Simplified API format: {simple_img.format}")

def main():
    """Main function to run all tests"""
    print("=== Image Upscaling API Test Suite ===")
    
    # Create test directory
    create_test_dir()
    
    # Create test image
    create_test_image()
    
    # Test Real-ESRGAN API
    realesrgan_success = test_realesrgan_api()
    
    # Test Simplified API
    simple_success = test_simple_api()
    
    # Compare results
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        compare_results()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Real-ESRGAN API: {'SUCCESS' if realesrgan_success else 'FAILED'}")
    print(f"Simplified API: {'SUCCESS' if simple_success else 'FAILED'}")
    
    if not realesrgan_success and simple_success:
        print("\nRECOMMENDATION: Use the simplified API (upscale_api_simple.py) for your environment.")
    elif realesrgan_success and not simple_success:
        print("\nRECOMMENDATION: Use the Real-ESRGAN API (upscale_api.py) for your environment.")
    elif realesrgan_success and simple_success:
        print("\nBoth APIs are working in your environment. Use Real-ESRGAN for better quality or the simplified API for better compatibility.")

if __name__ == "__main__":
    main()
