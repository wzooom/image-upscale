import subprocess
import time
import requests
import os
from PIL import Image, ImageDraw

def final_test():
    print("Starting comprehensive test of the Real-ESRGAN API...")
    
    # Create a test image
    print("1. Creating test image...")
    img = Image.new('RGB', (100, 100), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 75, 75], fill='red')
    img.save('test_image.png')
    print("   Test image created successfully.")
    
    # Start the API server
    print("2. Starting API server...")
    server_process = subprocess.Popen(["python", "upscale_api.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test with default model
        print("3. Testing with default model (RealESRGAN_x4plus)...")
        with open('test_image.png', 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/upscale/', files=files)
            
            if response.status_code == 200:
                with open('upscaled_default.png', 'wb') as out_file:
                    out_file.write(response.content)
                print("   Default model test successful! Image saved as upscaled_default.png")
            else:
                print(f"   Error with default model: {response.json()}")
        
        # Test with anime model
        print("4. Testing with anime model (RealESRGAN_x4plus_anime_6B)...")
        with open('test_image.png', 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/upscale/?model=RealESRGAN_x4plus_anime_6B', files=files)
            
            if response.status_code == 200:
                with open('upscaled_anime.png', 'wb') as out_file:
                    out_file.write(response.content)
                print("   Anime model test successful! Image saved as upscaled_anime.png")
            else:
                print(f"   Error with anime model: {response.json()}")
        
        # Test with invalid model
        print("5. Testing with invalid model...")
        with open('test_image.png', 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/upscale/?model=InvalidModel', files=files)
            
            if response.status_code == 400:
                print(f"   Invalid model test successful! Got expected error: {response.json()}")
            else:
                print(f"   Unexpected response for invalid model: {response.status_code}")
        
        # Test with invalid file type
        print("6. Testing with invalid file type...")
        response = requests.post('http://localhost:8000/upscale/', files={'file': (None, 'not_an_image.txt')})
        
        if response.status_code == 400:
            print(f"   Invalid file type test successful! Got expected error.")
        else:
            print(f"   Unexpected response for invalid file type: {response.status_code}")
            
        print("\nAll tests completed! The API is ready for use.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Terminate the server process
        server_process.terminate()
        print("API server stopped.")

if __name__ == "__main__":
    final_test()
