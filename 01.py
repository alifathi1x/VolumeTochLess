import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


# def for detect to landmark
def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

cap = cv2.VideoCapture(0)

# limited for volume
volume_min = 0
volume_max = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # hand detection
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # landmark beetween fingers
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # calculate distance beetween two fingers
            distance = calculate_distance(thumb, index)

            # draw green color for distance beetween fingers
            h, w, _ = frame.shape
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # distance for up or down the volume
            if distance < 0.1:
                # while fingers are close tp each other
                pyautogui.press('volumeup')
                print("Volume Up")
            elif distance > 0.25:
                # while fingers are away from each other
                pyautogui.press('volumedown')
                print("Volume Down")

    cv2.imshow("Hand Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
