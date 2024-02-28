import cv2
from mediapipe.python.solutions.hands import HandLandmark as HandLM
from api.handTracker import handTracker
from api.SimpleHand import SimpleHand
from api.HandsList import HandsList
import keyboard
import time
from api.Managers.CursorManager import CursorManager

RT = "Right"
LF = "Left"
canvas = [int, int]

def main():
    bgImage = cv2.imread(r'img.png')
    cap = cv2.VideoCapture(0)

    tracker = handTracker()

    start_time = time.time()
    handPositionListRT = []
    handPositionListLF = []

    handsList = HandsList()

    debugDraw = True  # Determine if to draw landmarks and label on screen
    mode = False

    while True:
        if cv2.waitKey(1) == ord('m'):
            if not mode:
                mode = True
            else:
                mode = False

        success, image = cap.read()
        # canvas = [image.shape[1], image.shape[0]]
        image = cv2.flip(image, 1)
        # bgImage = cv2.resize(bgImage, canvas)
        image = tracker.processImage(image, debugDraw)
        handsList.clear()
        if tracker.foundHands():
            for i in range(tracker.handsCount):
                # Get landmarks
                lmlist = tracker.landmarksFinder(i, debugDraw, image)
                # Get Classification
                side, score = tracker.classificationFinder(i, debugDraw, image, lmlist)
                # Create simple hand object
                detectedHand = SimpleHand(i, side, score, lmlist)
                # Add this hand found to the list
                handsList.addHand(detectedHand)

                if detectedHand.side == RT:
                    if detectedHand.isIndexFingerUp():
                        if mode is True:
                            cv2.circle(bgImage, (detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP),
                                                 detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP)), 10, (0, 0, 255),
                                       cv2.FILLED)
                            start_time = time.time()
                        else:
                            if len(handPositionListRT) != 4:
                                handPositionListRT.append(detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP))
                                handPositionListRT.append(detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP))
                            else:
                                cv2.line(bgImage, (handPositionListRT[0], handPositionListRT[1]),
                                         (handPositionListRT[2], handPositionListRT[3]), (0, 0, 255), 10, cv2.FILLED)
                                start_time = time.time()
                                handPositionListRT.remove(handPositionListRT[0])
                                handPositionListRT.remove(handPositionListRT[0])
                    else:
                        handPositionListRT.clear()
                else:
                    if detectedHand.isIndexFingerUp():
                        if mode is True:
                            cv2.circle(bgImage, (detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP),
                                                 detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP)), 10, (255, 0, 0),
                                       cv2.FILLED)
                            start_time = time.time()
                        else:
                            if len(handPositionListLF) != 4:
                                handPositionListLF.append(detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP))
                                handPositionListLF.append(detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP))
                            else:
                                cv2.line(bgImage, (handPositionListLF[0], handPositionListLF[1]),
                                         (handPositionListLF[2], handPositionListLF[3]), (255, 0, 0), 10, cv2.FILLED)
                                start_time = time.time()
                                handPositionListLF.remove(handPositionListLF[0])
                                handPositionListLF.remove(handPositionListLF[0])
                    else:
                        handPositionListLF.clear()
        bgImageCopy = bgImage.copy()
        cursorManager = CursorManager("cursorRight", (30,30), "cursorLeft", (30, 30))
        if handsList.hasRight():
            x, y = handsList.right.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
            cursorManager.displayCursor(bgImageCopy, x, y, "Right")
        if handsList.hasLeft():
            x, y = handsList.left.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
            cursorManager.displayCursor(bgImageCopy, x, y, "Left")
        if keyboard.is_pressed('b') or (time.time() - start_time) > 90:
            bgImageCopy = cv2.imread(r"img.png")
            bgImage = cv2.imread(r"img.png")

        cv2.imshow("Video", image)

        cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Canvas", bgImageCopy)
        closeKey = cv2.waitKey(1) & 0xFF
        if closeKey == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
