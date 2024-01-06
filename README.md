# Hand Tracking and Drawing with Mediapipe

This Python script utilizes the Mediapipe library to track hand landmarks in real-time through a webcam feed. It allows you to draw lines on a background image using the position of your hands.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)

You can install these dependencies using the following commands:

```bash
pip install opencv-python
pip install mediapipe
```

## Usage

1. Save an image as `img.png` in the same directory as the script. This image will be used as the background.

2. Run the script.

```bash
python script_name.py
```

3. A window will open showing the webcam feed with hand landmarks drawn on the image.

## Controls

- **'x' key:** Increase the line thickness.
- **'v' key:** Decrease the line thickness (minimum thickness is 1).
- **'Esc' key:** Close the program.

## Drawing on the Background Image

- The script tracks the tip of the index finger for both left and right hands.
- The left hand's drawing color is red, and the right hand's drawing color is blue.
- Press 'x' to increase line thickness and 'v' to decrease it.

## Important Notes

- Make sure to have proper lighting for better hand tracking results.
- To exit the program, press the 'Esc' key.

Feel free to experiment with different background images and hand movements to create interesting drawings!
