
import mediapipe as mp
import cv2
import mediapipe as mp
import pyautogui
# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing_utils = mp.solutions.drawing_utils

# Get screen resolution
screen_w, screen_h = pyautogui.size()
click_time = time.time()
prev_time = 0

# Create a resizable window
cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Mouse", 900, 700)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    click_detected = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks and connections
        drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark

        # Index fingertip (id 8)
        x8 = int(landmarks[8].x * w)
        y8 = int(landmarks[8].y * h)

        # Thumb tip (id 4)
        x4 = int(landmarks[4].x * w)
        y4 = int(landmarks[4].y * h)

        # Draw circles
        cv2.circle(frame, (x8, y8), 10, (0, 255, 255), -1)
        cv2.circle(frame, (x4, y4), 10, (0, 255, 0), -1)

        # Draw line between thumb and index
        cv2.line(frame, (x8, y8), (x4, y4), (255, 0, 0), 3)

        # Move mouse pointer
        screen_x = int((x8 / w) * screen_w)
        screen_y = int((y8 / h) * screen_h)
        pyautogui.moveTo(screen_x, screen_y)

        # Detect click gesture
        if abs(y8 - y4) < 40:
            if time.time() - click_time > 1:
                pyautogui.click()
                click_time = time.time()
                click_detected = True

    # Show "CLICK" text if clicked
    if click_detected:
        cv2.putText(frame, "CLICK", (x8 + 20, y8 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show window
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
