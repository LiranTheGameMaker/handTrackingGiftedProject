import cv2
from mediapipe.python.solutions.hands import HandLandmark as HandLM
from handTracker import handTracker
from SimpleHand import SimpleHand
from HandsList import HandsList
import keyboard

RT = "Right"
LF = "Left"
canvas = [int, int]


# def maskOutput(image, canvas_size : list):
# return maskList.process_image(image, canvas_size)


def main():
    bgImage = cv2.imread(r'img.png')
    cap = cv2.VideoCapture(1)
    tracker = handTracker()

    handPositionListRT = []
    handPositionListLF = []
    handsList = HandsList()
    debugDraw = True  # Determine if to draw landmarks and label on screen
    mode = False

    while True:
        if cv2.waitKey(1) & 0xFF == ord('m'):
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
                        else:
                            if len(handPositionListRT) != 4:
                                handPositionListRT.append(detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP))
                                handPositionListRT.append(detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP))
                            else:
                                cv2.line(bgImage, (handPositionListRT[0], handPositionListRT[1]),
                                         (handPositionListRT[2], handPositionListRT[3]), (0, 0, 255), 10, cv2.FILLED)
                                handPositionListRT.remove(handPositionListRT[0])
                                handPositionListRT.remove(handPositionListRT[0])

                else:
                    if detectedHand.isIndexFingerUp():
                        if mode is True:
                            cv2.circle(bgImage, (detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP),
                                                 detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP)), 10, (255, 0, 0),
                                       cv2.FILLED)
                        else:
                            if len(handPositionListLF) != 4:
                                handPositionListLF.append(detectedHand.getLandmarkX(HandLM.INDEX_FINGER_TIP))
                                handPositionListLF.append(detectedHand.getLandmarkY(HandLM.INDEX_FINGER_TIP))
                            else:
                                cv2.line(bgImage, (handPositionListLF[0], handPositionListLF[1]),
                                         (handPositionListLF[2], handPositionListLF[3]), (255, 0, 0), 10, cv2.FILLED)
                                handPositionListLF.remove(handPositionListLF[0])
                                handPositionListLF.remove(handPositionListLF[0])

        print(handsList)
        if keyboard.is_pressed('b'):
            bgImage = cv2.imread(r"img.png")
        cv2.imshow("Video", image)
        cv2.imshow("Canvas", bgImage)
        closeKey = cv2.waitKey(1) & 0xFF
        print(closeKey)
        if closeKey == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
