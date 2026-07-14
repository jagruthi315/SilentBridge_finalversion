import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

# ==========================================================
# SETTINGS
# ==========================================================

DATA_PATH = "Processed_Video_Dataset"

MODEL_NAME = "best_lstm_video.keras"

FINAL_MODEL = "lstm_video_model.keras"

LABEL_FILE = "label_classes_video.npy"

# ==========================================================
# LOAD DATASET
# ==========================================================

X = []

y = []

print("\n====================================")

print("Loading Dataset...")

print("====================================")

for label in sorted(os.listdir(DATA_PATH)):

    label_path = os.path.join(
        DATA_PATH,
        label
    )

    if not os.path.isdir(label_path):
        continue

    for file in sorted(os.listdir(label_path)):

        if not file.endswith(".npy"):
            continue

        sample = np.load(

            os.path.join(
                label_path,
                file
            )

        )

        X.append(sample)

        y.append(label)

X = np.array(X)

y = np.array(y)

X = X.astype(np.float32)

print("\nDataset Loaded Successfully")

print("Dataset Shape :", X.shape)

print("Total Samples :", len(X))

print("Total Classes :", len(np.unique(y)))

# ==========================================================
# VERIFY SHAPE
# ==========================================================

assert X.shape[1] == 60, "Sequence length should be 60"

assert X.shape[2] == 126, "Feature size should be 126"

print("\nInput Shape Verified ✔")
