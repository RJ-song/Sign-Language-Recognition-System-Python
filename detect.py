import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, img = cap.read()
        image = img.copy()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Sign Language Recognition System', image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
