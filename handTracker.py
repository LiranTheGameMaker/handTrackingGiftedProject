import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark as HandLM

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.handsCount = 0

    def foundHands(self):
        return self.handsCount > 0

    def processImage(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        count = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                count += 1
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        self.handsCount = count
        return image

    def landmarksFinder(self, handNo=0, draw=True, image=None):
        if handNo >= self.handsCount:
            return
        lmlist = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw and image is not None:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            return lmlist

    def classificationFinder(self, handNo=0, write=True, image=None, lmlist=None, targetLm=HandLM.WRIST):
        if self.results.multi_handedness:
            hand = self.results.multi_handedness[handNo]
            label = hand.classification[0].label
            score = hand.classification[0].score

            if write and image is not None:
                if lmlist is None:
                    lmlist = self.landmarksFinder(handNo, False)

                if len(lmlist) > targetLm:
                    # Write text on the image
                    font = cv2.FONT_HERSHEY_SIMPLEX  # Choose the font
                    font_scale = 1  # Font size
                    color = (255, 0, 0)  # Font color (B, G, R)
                    thickness = 2  # Font thickness

                    x = lmlist[targetLm][1]
                    y = lmlist[targetLm][2]
                    cv2.putText(image, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            return label, score

