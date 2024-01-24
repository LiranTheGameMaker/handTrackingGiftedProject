import cv2
from PIL import Image


# Define a function to determine if a pixel is white
def is_white(pixel, tolerance):
    return pixel >= (255 - tolerance)  # 255 corresponds to white in grayscale


def process_image(image, frame_size, tolerance=100):
    # Open the image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((frame_size[0], frame_size[1]))
    pil_img = pil_img.convert("L")
    # Get the pixel data
    pixels = list(pil_img.getdata())
    # Create the result list based on whether each pixel is white or not
    result = [0 if is_white(pixel, tolerance) else 1 for pixel in pixels]
    return result
