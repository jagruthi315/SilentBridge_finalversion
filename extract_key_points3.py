import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ==========================================================
# SETTINGS
# ==========================================================

DATASET_PATH = "Video_Dataset"

OUTPUT_FOLDER = "Processed_Video_Dataset"

MODEL_PATH = "hand_landmarker.task"

SEQUENCE_LENGTH = 60

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
# EXTRACT KEYPOINTS
# ==========================================================

def extract_keypoints(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )

    result = detector.detect(mp_image)

    left = [0.0] * 63
    right = [0.0] * 63

    if result.hand_landmarks:

        for landmarks, handedness in zip(

                result.hand_landmarks,

                result.handedness):

            coords = []

            for lm in landmarks:

                coords.extend([

                    lm.x,

                    lm.y,

                    lm.z

                ])

            label = handedness[0].category_name

            if label == "Left":

                left = coords

            else:

                right = coords

    return np.array(left + right)
# ==========================================================
# PROCESS VIDEO
# ==========================================================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    sequence = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        keypoints = extract_keypoints(frame)

        sequence.append(keypoints)

        if len(sequence) >= SEQUENCE_LENGTH:
            break

    cap.release()

    # ----------------------------------------
    # Pad with zeros if video has fewer frames
    # ----------------------------------------

    while len(sequence) < SEQUENCE_LENGTH:

        sequence.append(np.zeros(126))

    sequence = np.array(sequence)

    return sequence

    # ==========================================================
# BUILD DATASET
# ==========================================================

total_words = 0
total_videos = 0

for word in sorted(os.listdir(DATASET_PATH)):

    word_folder = os.path.join(
        DATASET_PATH,
        word
    )

    if not os.path.isdir(word_folder):
        continue

    total_words += 1

    save_folder = os.path.join(
        OUTPUT_FOLDER,
        word
    )

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    print("\n====================================")
    print(f"Processing : {word}")

    videos = sorted([

        x

        for x in os.listdir(word_folder)

        if x.lower().endswith(
            (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv"
            )
        )

    ])

    print(f"Videos Found : {len(videos)}")

    for video in videos:

        video_path = os.path.join(
            word_folder,
            video
        )

        save_name = os.path.splitext(video)[0] + ".npy"

        save_path = os.path.join(
            save_folder,
            save_name
        )

        # -----------------------------
        # Skip if already processed
        # -----------------------------

        if os.path.exists(save_path):

            print(f"Skipped : {save_name}")

            continue

        sequence = process_video(video_path)

        np.save(
            save_path,
            sequence
        )

        total_videos += 1

        print(f"Saved : {save_name}")
        # ==========================================================
# FINISHED
# ==========================================================

print("\n====================================")
print("Extraction Complete")
print("====================================")

print(f"Total Words Processed : {total_words}")

print(f"Total Videos Processed : {total_videos}")

print(f"\nOutput Folder : {OUTPUT_FOLDER}")

print("\nEach file shape : (60,126)")

print("\nFinished Successfully.")

detector.close()

print("\n====================================")
