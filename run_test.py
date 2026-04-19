import subprocess
import time
import requests
import os
import image_utils

def run_api_test():
    # First create a test image
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 75, 75], fill='red')
    
    # Save to the input directory
    output_path = image_utils.get_input_path('test_image.png')
    img.save(output_path)
    print(f"Test image created: {output_path}")
    
    # Start the API server in the background
    print("Starting API server...")
    server_process = subprocess.Popen(["python", "upscale_api.py"])
    
    # Wait a moment for the server to start
    time.sleep(3)
    
    try:
        # Test the API with the sample image
        test_image_path = image_utils.get_input_path('test_image.png')
        with open(test_image_path, 'rb') as f:
            files = {'file': (os.path.basename(test_image_path), f, 'image/png')}
            response = requests.post('http://localhost:8000/upscale/', files=files)
            
            if response.status_code == 200:
                output_path = image_utils.get_output_path('realesrgan', 'upscaled_image.png')
                with open(output_path, 'wb') as out_file:
                    out_file.write(response.content)
                print(f"Image upscaled successfully! Saved as {output_path}")
            else:
                print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Terminate the server process
        server_process.terminate()
        print("API server stopped.")

if __name__ == "__main__":
    run_api_test()
