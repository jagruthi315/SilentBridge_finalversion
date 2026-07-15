import os
import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "Video_Dataset"

NEW_VIDEOS_PER_WORD = 60

FPS = 30
SECONDS = 3

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Smaller size used ONLY for MediaPipe detection -> makes the loop fast
# so key presses (S / P / R / Q) respond instantly.
DETECT_WIDTH = 480
DETECT_HEIGHT = 270

RECORDER_NAME = "jagruthi"

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

# ============================================
# MEDIAPIPE TASKS
# ============================================

HAND_MODEL = "hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# Connections between hand landmark points (used to draw the skeleton lines)
# Hardcoded standard 21-point hand skeleton (mp.solutions is not available
# in newer mediapipe versions that use the Tasks API, so we define it ourselves).
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def delete_video(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print("Couldn't delete:", path)
        print(e)


def draw_text(frame, text, y, color, size=1, thickness=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


# ============================================
# LOAD WORDS
# ============================================

words = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

# ============================================
# RESUME FROM LAST COMPLETED WORD
# ============================================

print("\n====================================")
print("Resume Recording")
print("====================================")

last_word = input(
    "Enter LAST COMPLETED word (Press Enter to start from beginning): "
).strip()

if last_word:

    if last_word not in words:
        print(f"\n'{last_word}' not found in Video_Dataset!")
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        exit()

    start_index = words.index(last_word) + 1
    words = words[start_index:]

    if len(words) > 0:
        print(f"\nStarting from: {words[0]}")
    else:
        print("\nNo words left to record.")
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        exit()

print("\n====================================")
print(" SilentBridge Dataset Recorder ")
print("====================================")

print(f"\nTotal Words : {len(words)}")

print("\nKeyboard Shortcuts")
print("--------------------------")
print("ENTER -> Start current word")
print("S     -> Skip current video")
print("P     -> Skip current word (moves to next word immediately)")
print("R     -> Re-record current video")
print("Q     -> Quit")
print()

quit_all = False

# ============================================
# START RECORDING
# ============================================

for index, word in enumerate(words):

    if quit_all:
        break

    folder = os.path.join(DATASET_PATH, word)

    videos = sorted([x for x in os.listdir(folder) if x.endswith(".mp4")])
    existing = len(videos)

    print("\n" + "=" * 60)
    print(f"Word {index + 1}/{len(words)}")
    print("Current Word :", word)
    print("Already Present :", existing)
    print("Will Record :", NEW_VIDEOS_PER_WORD)
    print()
    print("Instructions")
    print("--------------------------")
    print("• Show the sign first")
    print("• Wait until hand is detected")
    print("• Recording starts automatically")
    print("• Press S to skip current video")
    print("• Press P to skip current word (instant)")
    print("• Press R to re-record current video")
    print("• Press Q anytime to quit")

    # ---- ask if frame count should be changed for this word ----
    default_total_frames = FPS * SECONDS

    frame_input = input(
        f"\nFrames to record per video for '{word}' "
        f"(default {default_total_frames} = {SECONDS} sec @ {FPS}fps)."
        f"\nPress ENTER to keep default, or type a new number: "
    ).strip()

    if frame_input:
        try:
            word_total_frames = int(frame_input)
            if word_total_frames <= 0:
                print("Invalid number, using default.")
                word_total_frames = default_total_frames
        except ValueError:
            print("Invalid input, using default.")
            word_total_frames = default_total_frames
    else:
        word_total_frames = default_total_frames

    print(f"-> Recording {word_total_frames} frames per video for '{word}'")

    input("\nPress ENTER when ready...")

    video_number = 0

    while video_number < NEW_VIDEOS_PER_WORD:

        print(f"\nPreparing Video {video_number + 1}/{NEW_VIDEOS_PER_WORD}")

        filename = os.path.join(
            folder,
            f"{RECORDER_NAME}_{existing + video_number + 1:03d}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        total_frames = word_total_frames
        recording = False
        frames_recorded = 0

        # action can be: None, "skip_video", "skip_word", "rerecord", "quit"
        action = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            # ---- fast detection on a SMALL resized copy ----
            # ---- HandLandmarker Detection ----

            small = cv2.resize(
                frame,
                (DETECT_WIDTH, DETECT_HEIGHT)
            )

            rgb_small = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_small
            )

            result = detector.detect(mp_image)

            hand_detected = False

            if result.hand_landmarks:
                hand_detected = True

                scale_x = FRAME_WIDTH / DETECT_WIDTH
                scale_y = FRAME_HEIGHT / DETECT_HEIGHT

                # ---- draw the hand skeleton (lines) + keypoints (dots) ----
                for landmarks in result.hand_landmarks:

                    # convert all landmarks to pixel coords first
                    points = []
                    for lm in landmarks:
                        cx = int(lm.x * DETECT_WIDTH * scale_x)
                        cy = int(lm.y * DETECT_HEIGHT * scale_y)
                        points.append((cx, cy))

                    # draw connection lines (white) first, so dots sit on top
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start_idx],
                            points[end_idx],
                            (255, 255, 255),
                            2
                        )

                    # draw landmark dots (red) on top of the lines
                    for (cx, cy) in points:
                        cv2.circle(
                            frame,
                            (cx, cy),
                            4,
                            (0, 0, 255),
                            -1
                        )

            # =====================================
            # WAIT FOR HAND / RECORD  (runs every frame, hand or not)
            # =====================================
            if not recording:
                draw_text(frame, "Waiting for Hand...", 40, (0, 255, 255))
                draw_text(frame, f"Word : {word}", 80, (255, 255, 255))
                draw_text(frame, f"Video : {video_number + 1}/{NEW_VIDEOS_PER_WORD}", 120, (255, 255, 255))
                draw_text(frame, "S=Skip Video  P=Skip Word", 160, (0, 255, 255), 0.7)
                draw_text(frame, "R=Re-record  Q=Quit", 195, (0, 255, 255), 0.7)

                if hand_detected:
                    print("Hand Detected -> Recording Started")
                    recording = True
            else:
                out.write(frame)
                frames_recorded += 1
                draw_text(frame, "RECORDING", 40, (0, 0, 255))
                draw_text(frame, f"{frames_recorded}/{total_frames}", 80, (0, 255, 0))
                draw_text(frame, word, 120, (255, 255, 255))

                if frames_recorded >= total_frames:
                    break

            cv2.imshow("Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("\nQuitting...")
                action = "quit"
                break

            elif key == ord("s"):
                print("\nSkipped current video.")
                action = "skip_video"
                break

            elif key == ord("p"):
                print(f"\nSkipping word : {word} (instant)")
                action = "skip_word"
                break

            elif key == ord("r"):
                print("\nRe-recording current video.")
                action = "rerecord"
                break

        # release writer once, cleanly, no matter how the inner loop ended
        out.release()

        if action == "quit":
            delete_video(filename)
            quit_all = True
            break

        elif action == "skip_video":
            delete_video(filename)
            video_number += 1
            continue

        elif action == "skip_word":
            delete_video(filename)
            print("Moved to next word.\n")
            break

        elif action == "rerecord":
            delete_video(filename)
            print("Recording the same video again...")
            continue

        else:
            print(f"Saved -> {filename}")
            video_number += 1

# ============================================
# ALL WORDS COMPLETED
# ============================================

print("\n====================================")
print("Dataset Recording Finished" if not quit_all else "Recording Stopped by User")
print("====================================")
detector.close()
cap.release()
cv2.destroyAllWindows()
