import requests

# Test the API with a sample image
def test_api():
    # Replace 'your_image.png' with the path to an actual test image
    with open('test_image.png', 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/upscale/', files=files)
        
        if response.status_code == 200:
            with open('upscaled_image.png', 'wb') as out_file:
                out_file.write(response.content)
            print("Image upscaled successfully!")
        else:
            print(f"Error: {response.json()}")

if __name__ == "__main__":
    test_api()
