import cv2
from mediapipe.python.solutions.hands import HandLandmark as HandLM
from handTracker import handTracker
from SimpleHand import SimpleHand
from HandsList import HandsList
import keyboard
import time
import numpy as np

RT = "Right"
LF = "Left"
canvas = [int, int]


def overlay_image(background, overlay, x_offset, y_offset):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = overlay.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = overlay[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = alpha_channel[:, :, np.newaxis]
    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    return background


def main():
    bgImage = cv2.imread(r'img.png')
    cursorRT = cv2.resize(cv2.imread(r'cursorRight.png', cv2.IMREAD_UNCHANGED), (30, 30))
    cursorLF = cv2.resize(cv2.imread(r'cursorLeft.png', cv2.IMREAD_UNCHANGED), (30, 30))
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
        print(handsList)
        if handsList.hasRight():
            x, y = handsList.right.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
            overlay_image(bgImageCopy, cursorRT, x, y)
        if handsList.hasLeft():
            x, y = handsList.left.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
            overlay_image(bgImageCopy, cursorLF, x, y)
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
