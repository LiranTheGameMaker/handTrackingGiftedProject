import cv2
from mediapipe.python.solutions.hands import HandLandmark as HandLM
from API.handTrackerWrapper import HandTrackerWrapper
from API.CursorManager import CursorManager


class SimpleDrawGame:
    def __init__(self):
        self.is_running = False

    def run(self):
        self.is_running = True
        bg_image = cv2.imread(r'Games/SimpleDrawGame/img.png')
        cursorManager = CursorManager(r'Games/SimpleDrawGame/cursorRight.png', r'Games/SimpleDrawGame/cursorRight.png')
        tracker = HandTrackerWrapper()
        hand_colors = {"Right": (255, 0, 0),
                       "Left": (0, 0, 255)}
        while True:
            tracker.update_hands_list()

            for hand in tracker.hands_list:
                if hand.isIndexFingerUp():
                    color = hand_colors[hand.side]
                    # draw circle
                    cv2.circle(bg_image, (hand.getLandmarkX(HandLM.INDEX_FINGER_TIP),
                                         hand.getLandmarkY(HandLM.INDEX_FINGER_TIP)), 10, color, cv2.FILLED)
            if tracker.hands_list.has_left():
                x, y = tracker.hands_list.left.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
                cursorManager.displayCursor(bg_image.copy(), x, y, "Left")
            if tracker.hands_list.has_right():
                x, y = tracker.hands_list.right.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
                cursorManager.displayCursor(bg_image.copy(), x, y, "Right")
            bg_image_copy = bg_image.copy()
            cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
            cv2.imshow("Canvas", bg_image_copy)

            image = tracker.get_hands_image()
            cv2.imshow("Video", image)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        cv2.destroyAllWindows()
        self.is_running = False

