import requests
from PIL import Image, ImageDraw
import io
import os
import time
import image_utils

def create_test_image():
    """Create a simple test image with a blue background and red square"""
    img = Image.new('RGB', (100, 100), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 75, 75], fill='red')
    
    # Save to the input directory
    output_path = image_utils.get_input_path('test_image.png')
    img.save(output_path)
    print(f"Test image created: {output_path}")
    return str(output_path)

def test_upscale_api(image_path, method='lanczos', scale=4):
    """Test the upscale API with the given image and parameters"""
    url = f"http://localhost:8000/upscale/?method={method}&scale={scale}"
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/png')}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            output_filename = f"upscaled_{method}_x{scale}.png"
            output_path = image_utils.get_output_path('simple', output_filename)
            
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

def run_tests():
    """Run a series of tests with different methods and scale factors"""
    # Create test image
    image_path = create_test_image()
    
    # Test different upscaling methods
    methods = ['lanczos', 'bicubic', 'bilinear', 'nearest']
    scales = [2, 4]
    
    results = []
    
    # Test root endpoint
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("API root endpoint test: SUCCESS")
            print(f"Response: {response.json()}")
        else:
            print(f"API root endpoint test: FAILED ({response.status_code})")
    except Exception as e:
        print(f"API root endpoint test: ERROR - {str(e)}")
    
    # Test upscaling with different methods and scales
    for method in methods:
        for scale in scales:
            print(f"\nTesting upscaling with method={method}, scale={scale}")
            success, output_path = test_upscale_api(image_path, method, scale)
            results.append({
                'method': method,
                'scale': scale,
                'success': success,
                'output_path': output_path
            })
    
    # Test invalid method
    print("\nTesting with invalid method")
    success, _ = test_upscale_api(image_path, 'invalid_method', 4)
    results.append({
        'method': 'invalid_method',
        'scale': 4,
        'success': success,
        'output_path': None
    })
    
    # Test invalid scale
    print("\nTesting with invalid scale")
    success, _ = test_upscale_api(image_path, 'lanczos', 10)
    results.append({
        'method': 'lanczos',
        'scale': 10,
        'success': success,
        'output_path': None
    })
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    success_count = sum(1 for r in results if r['success'])
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    return results

if __name__ == "__main__":
    run_tests()
