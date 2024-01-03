import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
bgimg = cv2.imread("img.png")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
hand_positions = {}

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # Create a copy of the background image for each frame
    bgimg_copy = bgimg.copy()

    if results.multi_hand_landmarks:
        for handID, handLms in enumerate(results.multi_hand_landmarks):
            hand_vector = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:  # Tip of the index finger for both left and right hands
                    hand_vector.extend([cx, cy])
                    hand_positions.setdefault(handID, []).append(hand_vector)
                    if handID == 0:  # Left hand
                        color = (0, 0, 255)  # Red
                    else:  # Right hand
                        color = (255, 0, 0)  # Blue
                    cv2.circle(image, (cx, cy), 25, color, cv2.FILLED)
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    # Draw lines connecting current positions to previous positions on the copied background image
    for handID, positions in hand_positions.items():
        if len(positions) > 1:
            for i in range(1, len(positions)):
                if handID == 0:
                    color = (0, 0, 255)  # Red for left hand
                else:
                    color = (255, 0, 0)  # Blue for right hand
                cv2.line(bgimg_copy, (positions[i - 1][0], positions[i - 1][1]), (positions[i][0], positions[i][1]),
                         color=color, thickness=2)

    cv2.imshow("Output", image)
    cv2.imshow("background image", bgimg_copy)
    cv2.waitKey(1)
