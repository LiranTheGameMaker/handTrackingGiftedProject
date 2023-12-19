import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
bgimg = cv2.imread(r'C:\Users\User\PycharmProjects\handTracking\img.png')
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
hand_postitions = []

while True:
    hand_vector = []
    success, image = cap.read()
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 9:
                    hand_vector.append(cx)
                    hand_vector.append(cy)
                    hand_postitions.append(hand_vector)
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    if len(hand_postitions) == 2:
        print(f'{hand_postitions[0][0]}, {hand_postitions[0][1]} |||||| {hand_postitions[1][0]}, {hand_postitions[1][1]}')
        cv2.line(bgimg, (hand_postitions[0][0], hand_postitions[0][1]), (hand_postitions[1][0], hand_postitions[1][1]), color=(255, 0, 255), thickness=2)
        hand_vector.clear()
        hand_postitions.clear()
    cv2.imshow("Output", image)
    cv2.imshow("background image", bgimg)
    cv2.waitKey(1)
