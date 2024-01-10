import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark as HandLM

RT = "Right"
LF = "Left"
class HandsList():
    def __int__(self):
        self.right = None
        self.left = None

    def __str__(self):
        result = ""
        if self.hasRight():
            result += str(self.right)
        else:
            result += "Right: not found. "
        if self.hasLeft():
            result += " " + str(self.left)
        else:
            result += "Left: not found. "
        return result


    def addHand(self, hand):
        if hand.side == RT:
            self.right = hand
        else:
            self.left = hand

    def count(self):
        return int(self.right is not None) + int(self.left is not None)

    def clear(self):
        self.left = None
        self.right = None

    def hasRight(self):
        return self.right is not None

    def hasLeft(self):
        return self.left is not None

class SimpleHand:
    def __init__(self, id, side, score, lmlist ):
        self.id = id
        self.side = side
        self.score = score
        self.lmlist = lmlist
        self.cx, self.cy = self.getLandmarkXY(HandLM.INDEX_FINGER_TIP)

    #Override for print purpse
    def __str__(self):
        return f"{self.side}:index position {self.cx}, {self.cy}"

    def getLandmarkXY(self, targetLm):
        x = self.lmlist[targetLm][1]
        y = self.lmlist[targetLm][2]
        return x, y

    #TBD
    def isIndexFingerUp(self):
        return False

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

    def landmarksFinder(self, handNo=0, draw=True, image = None):
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

    def classificationFinder(self, handNo=0, write=True, image = None, lmlist=None, targetLm = HandLM.WRIST):
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

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    handsList  = HandsList()
    debugDraw = True  #Determine if to draw landmarks and label on screen

    while True:
        success, image = cap.read()
        image = tracker.processImage(image, debugDraw)

        handsList.clear()
        if tracker.foundHands():
            for i in range(tracker.handsCount):
                # Get landmarks
                lmlist = tracker.landmarksFinder(i, debugDraw, image )
                # Get Classification
                side, score = tracker.classificationFinder(i, debugDraw, image, lmlist)
                # Create simple hand object
                detectedHand = SimpleHand(i, side, score, lmlist)
                # Add this hand found to the list
                handsList.addHand(detectedHand)

        print(handsList)

        cv2.imshow("Video", image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
