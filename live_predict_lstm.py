import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model("lstm_video_model.keras")

labels = np.load("label_classes_video.npy")

# =====================================
# MEDIAPIPE TASKS
# =====================================

MODEL_PATH = "hand_landmarker.task"

options = vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    num_hands=2,

    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

sequence = []

SEQUENCE_LENGTH = 30

# =====================================
# LOOP
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

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

            # Draw landmarks

            h, w, _ = frame.shape

            for lm in landmarks:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0,255,0),
                    -1
                )

    keypoints = np.array(left + right)

    sequence.append(keypoints)

    sequence = sequence[-SEQUENCE_LENGTH:]

    prediction = ""

    confidence = 0

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(sequence, axis=0)

        pred = model.predict(
            input_data,
            verbose=0
        )[0]

        prediction_index = np.argmax(pred)

        prediction = labels[prediction_index]

        confidence = pred[prediction_index]

    cv2.rectangle(
        frame,
        (0,0),
        (640,60),
        (0,0,0),
        -1
    )

    cv2.putText(
        frame,
        f"{prediction} ({confidence:.2f})",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "SilentBridge Live Prediction",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
