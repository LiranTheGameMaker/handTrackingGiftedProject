import cv2
from mediapipe.python.solutions.hands import HandLandmark as HandLM
from API.handTrackerWrapper import HandTrackerWrapper
from API.CursorManager import CursorManager
from API.QuickDrawPredictor import QuickDrawPredictor
import time
from tensorflow import keras
import matplotlib.pyplot as plt

class QuickDrawGame:
    def __init__(self):
        self.is_running = False

    def run(self):
        self.is_running = True
        start_time = time.time()
        model = keras.models.load_model(r'Games\QuickDrawGame\model.keras', compile=False)
        bg_image = cv2.resize(cv2.imread(r'Games/QuickDrawGame/img.png').copy(), (920, 768))
        cursorManager = CursorManager(r'Games/QuickDrawGame/cursorRight.png', r'Games/SimpleDrawGame/cursorRight.png')
        predictor = QuickDrawPredictor(model=model)
        tracker = HandTrackerWrapper()
        mode = 0
        handPositionListRT = []
        handPositionListLF = []
        hand_colors = {"Right": (0, 0, 0),
                       "Left": (0, 0, 0)}
        while True:
            tracker.update_hands_list()

            for hand in tracker.hands_list:
                if hand.isIndexFingerUp():
                    color = hand_colors[hand.side]
                    if mode == 1:
                        cv2.circle(bg_image, (hand.getLandmarkX(HandLM.INDEX_FINGER_TIP),
                                             hand.getLandmarkY(HandLM.INDEX_FINGER_TIP)), 6, color, cv2.FILLED)
                        start_time = time.time()
                    else:
                        if hand == tracker.hands_list.right:
                            if len(handPositionListRT) != 4:
                                handPositionListRT.append(hand.getLandmarkX(HandLM.INDEX_FINGER_TIP))
                                handPositionListRT.append(hand.getLandmarkY(HandLM.INDEX_FINGER_TIP))
                            else:
                                cv2.line(bg_image, (handPositionListRT[0], handPositionListRT[1]),
                                        (handPositionListRT[2], handPositionListRT[3]), color, 10, cv2.FILLED)
                                start_time = time.time()
                                handPositionListRT.remove(handPositionListRT[0])
                                handPositionListRT.remove(handPositionListRT[0])
                        else:
                            if len(handPositionListLF) != 4:
                                handPositionListLF.append(hand.getLandmarkX(HandLM.INDEX_FINGER_TIP))
                                handPositionListLF.append(hand.getLandmarkY(HandLM.INDEX_FINGER_TIP))
                            else:
                                cv2.line(bg_image, (handPositionListLF[0], handPositionListLF[1]),
                                        (handPositionListLF[2], handPositionListLF[3]), color, 10, cv2.FILLED)
                                start_time = time.time()
                                handPositionListLF.remove(handPositionListLF[0])
                                handPositionListLF.remove(handPositionListLF[0])
                if hand.isHandOpen():
                    plt.show()
            bg_image_copy = bg_image.copy()
            if tracker.hands_list.has_left():
                x, y = tracker.hands_list.left.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
                cursorManager.displayCursor(bg_image_copy, x, y, "Left")
            if tracker.hands_list.has_right():
                x, y = tracker.hands_list.right.getLandmarkXY(HandLM.INDEX_FINGER_TIP)
                cursorManager.displayCursor(bg_image_copy, x, y, "Right")
            if (time.time() - start_time) >= 2:
                prediction = predictor.process(predictor.predict(bg_image.copy()))
                print(prediction)
                start_time = time.time()
                #cv2.putText(bg_image, predictor.label_list[prediction[0]])'

            cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
            cv2.imshow("Canvas", bg_image_copy)

            image = tracker.get_hands_image()
            cv2.imshow("Video", image)

            if (cv2.waitKey(1) & 0xFF) == ord('m'):
                plt.show()
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        cv2.destroyAllWindows()
        self.is_running = False

