import cv2
import mediapipe as mp
import numpy as np
from Hand import *

cap = cv2.VideoCapture(0)
bgimg = cv2.imread(r'C:\Users\User\PycharmProjects\wotor\img.png')
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hand_postitions = []
hands_list = []
myhandslist = [Hand(), Hand()]

def update_hands(results):
    for idx, (classification, landmarks) in enumerate(zip(results.multi_handedness, results.multi_hand_landmarks)):
        myhandslist[idx].Update(classification, landmarks)

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mpHands.HandLandmark.WRIST].x, hand.landmark[mpHands.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))

            output = text, coords

    return output


with mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        hand_vector = []
        success, frame = cap.read()
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)
        update_hands(results)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):  # working with each hand for landmark print
                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)  # adding text representing which hand is present

                for id, lm in enumerate(hand.landmark):  # giving an id to each landmark
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # defining X and Y coordinates for each landmark
                    if id == 9:  # checking if id of landmark is 9
                        hand_vector.append(cx)  # adding x and y coords to a list
                        hand_vector.append(cy)

                        for num, handInFrame in enumerate(results.multi_hand_landmarks):
                            if handInFrame == hand:
                                if get_label(num, hand, results):
                                    text, coord = get_label(num, hand, results)
                                    if not any(isinstance(i, str) for i in hands_list):  # check if there isnt an item of type string in the list
                                        hands_list.append(text)
                        hands_list.append(hand_vector)
                        if len(hands_list) == 3:
                            print(f"{hands_list} - hands list")
                            hand_postitions.append(hands_list)
                            hands_list.clear()
                        cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)  # marking the landmark with the id 9
                    mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS)  # drawing the landmarks to the screen

            if len(hand_postitions) != 0:
                print(f"{hand_postitions} - hand positions")
                hand_postitions.clear()



        # if len(hand_postitions) == 2:
        #    print(f'{hand_postitions[0][0]}, {hand_postitions[0][1]} |||||| {hand_postitions[1][0]}, {hand_postitions[1][1]}')
        #    cv2.line(bgimg, (hand_postitions[0][0], hand_postitions[0][1]), (hand_postitions[1][0], hand_postitions[1][1]), color=(255, 0, 255), thickness=2)
        #    hand_vector.clear()
        #    hand_postitions.clear()
        cv2.imshow("Output", image)
        cv2.imshow("background image", bgimg)
        cv2.waitKey(1)

