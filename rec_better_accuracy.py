import os
import cv2
import mediapipe as mp

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "Video_Dataset"

NEW_VIDEOS_PER_WORD = 20

FPS = 30
SECONDS = 2

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ============================================
# MEDIAPIPE
# ============================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ============================================
# LOAD WORDS
# ============================================

words = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("\n====================================")
print(" SilentBridge Dataset Recorder ")
print("====================================")

print(f"\nTotal Words : {len(words)}")

# ============================================

for index, word in enumerate(words):

    folder = os.path.join(DATASET_PATH, word)

    videos = sorted([
        x
        for x in os.listdir(folder)
        if x.endswith(".mp4")
    ])

    existing = len(videos)

    print("\n" + "=" * 60)

    print(f"Word {index+1}/{len(words)}")

    print("Current Word :", word)

    print("Already Present :", existing)

    print("Will Record :", NEW_VIDEOS_PER_WORD)

    print()

    print("Instructions")

    print("--------------------------")

    print("• Show the sign first")

    print("• Wait until hand is detected")

    print("• Recording starts automatically")

    print("• Press Q anytime to quit")

    input("\nPress ENTER when ready...")

    # ============================================
    # RECORD EACH VIDEO
    # ============================================

    for video_number in range(NEW_VIDEOS_PER_WORD):

        print(f"\nPreparing Video {video_number+1}/{NEW_VIDEOS_PER_WORD}")

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

        total_frames = FPS * SECONDS

        recording = False

        frames_recorded = 0
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            hand_detected = False

            if results.multi_hand_landmarks:

                hand_detected = True

                for hand_landmarks in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # ---------------------------------
            # WAIT FOR HAND
            # ---------------------------------

            if not recording:

                cv2.putText(
                    frame,
                    "Waiting for Hand...",
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

                    print("Hand Detected -> Recording Started")

                    recording = True

            # ---------------------------------
            # RECORD
            # ---------------------------------

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

            cv2.imshow("Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:

                print("\nRecording Cancelled.")

                out.release()

                cap.release()

                cv2.destroyAllWindows()

                exit()

        out.release()

        print(f"Saved -> {filename}")

print("\n====================================")
print("Dataset Recording Finished")
print("====================================")

hands.close()

cap.release()

cv2.destroyAllWindows()
