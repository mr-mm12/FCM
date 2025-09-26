from pystray import Icon, MenuItem, Menu
from screeninfo import get_monitors
from PIL import Image, ImageDraw
import mediapipe as mp
import numpy as np
import pyautogui
import time
import cv2
import sys


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Get screen resolution for controlling mouse
screen_width, screen_height = pyautogui.size()

# Move mouse to the center of the screen at program start
pyautogui.moveTo(screen_width // 2, screen_height // 2)

# Store initial eye positions for calibration
initial_left_eye_x, initial_left_eye_y = None, None
initial_right_eye_x, initial_right_eye_y = None, None

# Track timing for head movement (up, down, left, right) and no-face detection
look_down_time = 0
look_up_time = 0
look_left_time = 0
look_right_time = 0
no_face_time = None

# Track eye-closed timing for left/right click detection
left_eye_closed_start = None
left_eye_clicked = False
right_click_start = None


def create_image():
    """Create a simple tray icon (black square)."""
    image = Image.new('RGBA', (64, 64), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 64, 64), fill=(0, 0, 0))
    return image


def show_instructions():
    """Display instructions window before starting the program."""
    width, height = 940, 780
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    text_s = "---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_---_"
    text1 = "1_ Move your head in different directions to move the mouse."
    text2 = "Hold your hand in front of the camera to exit the app."
    text3 = "2_ Closing your eyes performs a left-click."
    text4 = "Keeping them closed longer performs a right-click."
    text5 = "Press any key to continue."

    cv2.putText(frame, text1, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text_s, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text3, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text4, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 168, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text5, (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (0, 128, 0), 2, cv2.LINE_AA)

    cv2.imshow("Instructions", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_quit(icon, item):
    """Quit program and release resources."""
    icon.stop()
    cap.release()
    cv2.destroyAllWindows()
    exit()


def show_center_screen():
    """Show a guide asking user to look at the center of the screen."""
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (0, 255, 255)

    center_x, center_y = width // 2, height // 2
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height

    circle_radius = 50
    cv2.circle(frame, (center_x, center_y), circle_radius, (255, 0, 255), 2)
    cv2.putText(frame, "Look at the center of the screen", 
                (center_x - 250, center_y - circle_radius - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("Center Screen", cv2.WINDOW_NORMAL)
    window_x = (screen_width - width) // 2
    window_y = (screen_height - height) // 2
    cv2.moveWindow("Center Screen", window_x, window_y)
    cv2.imshow("Center Screen", frame)

    start_time = time.time()
    while True:
        if time.time() - start_time >= 5:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to skip
            break
    cv2.destroyAllWindows()


# Show help and calibration screen before starting
show_instructions()
show_center_screen()

# Run MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    menu = Menu(MenuItem('Quit', on_quit))
    icon = Icon("Eye Control", create_image(), menu=menu)
    icon.run_detached()

    face_detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Frame dimensions
        height, width, _ = frame.shape

        # Draw calibration oval on screen center
        center_x, center_y = width // 2, height // 2
        ellipse_width, ellipse_height = 150, 215
        cv2.ellipse(frame, (center_x, center_y), (ellipse_width, ellipse_height), 
                    0, 0, 360, (0, 255, 0), 2)

        # Process face landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Get nose/face center landmark
                face_center = face_landmarks.landmark[168]
                face_x, face_y = int(face_center.x * width), int(face_center.y * height)

                # Check if face center is inside the oval (calibration area)
                dx = (face_x - center_x) / ellipse_width
                dy = (face_y - center_y) / ellipse_height
                if dx**2 + dy**2 <= 1:
                    face_detected = True
                    break
                else:
                    face_detected = False

        # Warn user if face is not aligned in oval
        if not face_detected:
            cv2.putText(frame, "Please align your face inside the oval", 
                        (width // 2 - 200, height // 2 + 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                cv2.destroyAllWindows()
                break
            continue

        # Reset warning once face is detected
        if face_detected:
            cv2.destroyAllWindows()

        # Track time since last face detection
        if not results.multi_face_landmarks:
            if no_face_time is None:
                no_face_time = time.time()
        else:
            no_face_time = None

        # Exit if no face detected for more than 0.3 sec
        if no_face_time and time.time() - no_face_time > 0.3:
            icon.visible = False
            icon.stop()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

        current_time = time.time()

        # Get eye landmark positions
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[133]
        left_eye_x, left_eye_y = int(left_eye.x * width), int(left_eye.y * height)
        right_eye_x, right_eye_y = int(right_eye.x * width), int(right_eye.y * height)

        # Save initial positions (calibration)
        if initial_left_eye_x is None and initial_right_eye_x is None:
            initial_left_eye_x, initial_left_eye_y = left_eye_x, left_eye_y
            initial_right_eye_x, initial_right_eye_y = right_eye_x, right_eye_y

        # Calculate head movement deltas
        delta_x_left = left_eye_x - initial_left_eye_x
        delta_y_left = left_eye_y - initial_left_eye_y
        delta_x_right = right_eye_x - initial_right_eye_x
        delta_y_right = right_eye_y - initial_right_eye_y

        threshold = 12  # Minimum movement to trigger actions
        move_distance_after2 = 60  # Faster movement after holding direction

        # Vertical mouse movement (up/down)
        vertical_move_distance = 15
        if delta_y_left > threshold and delta_y_right > threshold:
            if look_down_time == 0:
                look_down_time = current_time
            elif current_time - look_down_time >= 2:
                vertical_move_distance = move_distance_after2
            pyautogui.move(0, vertical_move_distance)
        elif delta_y_left < -threshold and delta_y_right < -threshold:
            if look_up_time == 0:
                look_up_time = current_time
            elif current_time - look_up_time >= 2:
                vertical_move_distance = move_distance_after2
            pyautogui.move(0, -vertical_move_distance)
        else:
            look_down_time = 0
            look_up_time = 0

        # Horizontal mouse movement (left/right)
        horizontal_move_distance = 20
        if delta_x_left < -threshold and delta_x_right < -threshold:
            if look_left_time == 0:
                look_left_time = current_time
            elif current_time - look_left_time >= 2:
                horizontal_move_distance = move_distance_after2
            pyautogui.move(-horizontal_move_distance, 0)
        elif delta_x_left > threshold and delta_x_right > threshold:
            if look_right_time == 0:
                look_right_time = current_time
            elif current_time - look_right_time >= 2:
                horizontal_move_distance = move_distance_after2
            pyautogui.move(horizontal_move_distance, 0)
        else:
            look_left_time = 0
            look_right_time = 0

        # Eye Aspect Ratio (EAR) for blink detection
        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        vertical_distance_left = abs(left_eye_top.y - left_eye_bottom.y) * height
        horizontal_distance_left = abs(face_landmarks.landmark[33].x - face_landmarks.landmark[133].x) * width
        left_eye_aspect_ratio = vertical_distance_left / horizontal_distance_left

        EAR_THRESHOLD = 0.3  # Smaller = eyes closed

        # Left-click if eye closed for 0.3s
        if left_eye_aspect_ratio < EAR_THRESHOLD:
            if left_eye_closed_start is None:
                left_eye_closed_start = current_time
            elif current_time - left_eye_closed_start >= 0.3 and not left_eye_clicked:
                pyautogui.click()
                left_eye_clicked = True
        else:
            left_eye_closed_start = None
            left_eye_clicked = False

        # Right-click if eye closed for 0.7s
        if left_eye_aspect_ratio < EAR_THRESHOLD:
            if right_click_start is None:
                right_click_start = current_time
            elif current_time - right_click_start >= 0.7:
                pyautogui.click(button='right')
                right_click_start = None
        else:
            right_click_start = None

    cap.release()
    cv2.destroyAllWindows()
