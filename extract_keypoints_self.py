import os
import cv2
import time
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ==========================================================
# SETTINGS
# ==========================================================

DATASET_PATH = "Video_Dataset"

NEW_VIDEOS_PER_WORD = 20

MODEL_PATH = "hand_landmarker.task"

FPS = 30
SECONDS = 2

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ==========================================================
# MEDIAPIPE TASKS
# ==========================================================

options = vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    num_hands=2,

    running_mode=vision.RunningMode.IMAGE

)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================================================
# DRAW LANDMARKS
# ==========================================================

HAND_CONNECTIONS = [

    (0,1),(1,2),(2,3),(3,4),

    (0,5),(5,6),(6,7),(7,8),

    (5,9),(9,10),(10,11),(11,12),

    (9,13),(13,14),(14,15),(15,16),

    (13,17),(17,18),(18,19),(19,20),

    (0,17)

]

# ==========================================================
# DETECT HAND
# ==========================================================

def detect_hand(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    return result


# ==========================================================
# DRAW LANDMARKS
# ==========================================================

def draw_landmarks(frame, result):

    if not result.hand_landmarks:
        return frame

    h, w, _ = frame.shape

    for landmarks in result.hand_landmarks:

        points = []

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                4,
                (0,255,0),
                -1
            )

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255,0,0),
                2
            )

    return frame


# ==========================================================
# LOAD WORDS
# ==========================================================

words = sorted([

    folder

    for folder in os.listdir(DATASET_PATH)

    if os.path.isdir(os.path.join(DATASET_PATH, folder))

])

print("\n====================================")
print(" SilentBridge Recorder ")
print("====================================")

print(f"\nTotal Words : {len(words)}")

# ==========================================================
# MAIN LOOP
# ==========================================================

for index, word in enumerate(words):

    folder = os.path.join(DATASET_PATH, word)

    existing = len([

        x

        for x in os.listdir(folder)

        if x.lower().endswith(".mp4")

    ])

    print("\n====================================")

    print(f"Word {index+1}/{len(words)}")

    print("Current Word :", word)

    print("Existing Videos :", existing)

    print("Will Record :", NEW_VIDEOS_PER_WORD)

    input("\nPress ENTER to continue...")

    for video_number in range(NEW_VIDEOS_PER_WORD):

        filename = os.path.join(

            folder,

            f"jagruthi_{existing+video_number+1:03d}.mp4"

        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(

            filename,

            fourcc,

            FPS,

            (FRAME_WIDTH, FRAME_HEIGHT)

        )

        recording = False

        frames_recorded = 0

        total_frames = FPS * SECONDS
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            result = detect_hand(frame)

            frame = draw_landmarks(frame, result)

            hand_detected = (
                result.hand_landmarks is not None and
                len(result.hand_landmarks) > 0
            )

            # -----------------------------------------
            # WAIT FOR HAND
            # -----------------------------------------

            if not recording:

                cv2.putText(
                    frame,
                    "Show Gesture",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Word : {word}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Video : {video_number+1}/{NEW_VIDEOS_PER_WORD}",
                    (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2
                )

                if hand_detected:

                    recording = True

                    print(f"\nRecording started -> {word}")

                    time.sleep(0.25)

            # -----------------------------------------
            # RECORD
            # -----------------------------------------

            else:

                out.write(frame)

                frames_recorded += 1

                cv2.putText(
                    frame,
                    "RECORDING",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"{frames_recorded}/{total_frames}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    word,
                    (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2
                )

                if frames_recorded >= total_frames:

                    break

            cv2.imshow("SilentBridge Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:

                print("\nRecording cancelled.")

                out.release()

                detector.close()

                cap.release()

                cv2.destroyAllWindows()

                exit()

        out.release()

        print(f"Saved -> {os.path.basename(filename)}")

print("\n==========================================")
print("Recording Finished")
print("==========================================")

detector.close()

cap.release()


cv2.destroyAllWindows()
