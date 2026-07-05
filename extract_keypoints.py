import os
import cv2
import numpy as np

import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "Video_Dataset"
OUTPUT_PATH = "Processed_Video_Dataset"

SEQUENCE_LENGTH = 30

MODEL_PATH = "hand_landmarker.task"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================
# MEDIAPIPE TASKS
# ==========================================

options = vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    num_hands=2,

    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# EXTRACT KEYPOINTS
# ==========================================

def extract_frame_keypoints(frame):

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

    return left + right


# ==========================================
# PROCESS VIDEO
# ==========================================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    sequence = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        keypoints = extract_frame_keypoints(frame)

        sequence.append(keypoints)

    cap.release()

    sequence = np.array(sequence)

    # Pad if video has fewer than 30 frames
    if len(sequence) < SEQUENCE_LENGTH:

        padding = np.zeros(
            (
                SEQUENCE_LENGTH - len(sequence),
                126
            )
        )

        sequence = np.vstack([sequence, padding])

    # Keep first 30 frames
    else:

        sequence = sequence[:SEQUENCE_LENGTH]

    return sequence


# ==========================================
# MAIN
# ==========================================

for label in os.listdir(DATASET_PATH):

    class_folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_folder):
        continue

    output_folder = os.path.join(
        OUTPUT_PATH,
        label
    )

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {label}")

    for video in os.listdir(class_folder):

        if not video.lower().endswith(
                (".mp4", ".avi", ".mov", ".MOV")):
            continue

        video_path = os.path.join(
            class_folder,
            video
        )

        save_name = os.path.splitext(video)[0] + ".npy"

        save_path = os.path.join(
            output_folder,
            save_name
        )

        # ======================================
        # SKIP IF ALREADY PROCESSED
        # ======================================

        if os.path.exists(save_path):
            print(f"Skipping: {save_name}")
            continue

        sequence = process_video(video_path)

        np.save(
            save_path,
            sequence
        )

        print(f"Saved: {save_name}")

print("\nFinished extracting all videos!")
