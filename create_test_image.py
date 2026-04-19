from PIL import Image, ImageDraw
import numpy as np
import image_utils

# Create a simple test image
def create_test_image():
    # Create a 100x100 image with a blue background and a red square
    img = Image.new('RGB', (100, 100), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 75, 75], fill='red')
    
    # Save the image to the input directory
    output_path = image_utils.get_input_path('test_image.png')
    img.save(output_path)
    print(f"Test image created: {output_path}")

if __name__ == "__main__":
    create_test_image()
