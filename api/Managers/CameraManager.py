import cv2
from mediapipe.python.solutions.hands import HandLandmark as HandLM
from api.handTracker import handTracker
from api.SimpleHand import SimpleHand
from api.HandsList import HandsList
import keyboard
import time

class CameraManager():
    def __init__(self):
        self.handsList = HandsList()
    def run(self):
        cap = cv2.VideoCapture(0)
        tracker = handTracker()
        debugDraw = False  # Determine if to draw landmarks and label on screen

        while True:
            success, image = cap.read()
            image = cv2.flip(image, 1)
            image = tracker.processImage(image, debugDraw)

            self.handsList.clear()
            if tracker.foundHands():
                for i in range(tracker.handsCount):
                    # Get landmarks
                    lmlist = tracker.landmarksFinder(i, debugDraw, image)
                    # Get Classification
                    side, score = tracker.classificationFinder(i, debugDraw, image, lmlist)
                    # Create simple hand object
                    detectedHand = SimpleHand(i, side, score, lmlist)
                    # Add this hand found to the list
                    self.handsList.addHand(detectedHand)



