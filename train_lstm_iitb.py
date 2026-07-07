# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import os

# DATA_PATH = "Processed_Video_Dataset"

# X = []
# y = []

# print("Loading dataset...")

# for label in os.listdir(DATA_PATH):

#     label_path = os.path.join(DATA_PATH, label)

#     if not os.path.isdir(label_path):
#         continue

#     for file in os.listdir(label_path):

#         if file.endswith(".npy"):

#             sample = np.load(os.path.join(label_path, file))

#             X.append(sample)
#             y.append(label)

# X = np.array(X)
# y = np.array(y)

# print("Dataset Loaded")
# print("Shape:", X.shape)

# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# num_classes = len(np.unique(y))

# y = to_categorical(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=np.argmax(y, axis=1)
# )

# print("Building model...")

# model = Sequential()

# model.add(LSTM(128, return_sequences=True, input_shape=(30,126)))
# model.add(Dropout(0.3))

# model.add(LSTM(64))
# model.add(Dropout(0.3))

# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.3))

# model.add(Dense(num_classes, activation="softmax"))

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=10,
#     restore_best_weights=True
# )

# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_test, y_test),
#     epochs=100,
#     batch_size=16,
#     callbacks=[early_stop]
# )

# loss, accuracy = model.evaluate(X_test, y_test)

# print("\nFinal Accuracy:", accuracy)

# # model.save("lstm_iitb_model.keras")

# # np.save("label_classes_iitb.npy", encoder.classes_)
# model.save("lstm_video_model.keras")

# np.save(
#     "label_classes_video.npy",
#     encoder.classes_
# )

# print("\nModel Saved Successfully!")


import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam

# ==========================================================
# DATASET
# ==========================================================

DATA_PATH = "Processed_Video_Dataset"

X = []
y = []

print("Loading dataset...")

for label in os.listdir(DATA_PATH):

    label_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):

        if file.endswith(".npy"):

            sample = np.load(
                os.path.join(label_path, file)
            )

            X.append(sample)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("\nDataset Loaded")
print("Shape :", X.shape)
print("Samples :", len(X))
print("Classes :", sorted(set(y)))

# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))

y = to_categorical(y_encoded)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================================
# MODEL
# ==========================================================

print("\nBuilding model...")

model = Sequential()

model.add(
    LSTM(
        256,
        return_sequences=True,
        input_shape=(30,126)
    )
)

model.add(BatchNormalization())
model.add(Dropout(0.25))

# ----------------------------------------------------------

model.add(
    LSTM(
        128,
        return_sequences=True
    )
)

model.add(BatchNormalization())
model.add(Dropout(0.25))

# ----------------------------------------------------------

model.add(LSTM(64))
model.add(Dropout(0.25))

# ----------------------------------------------------------

model.add(Dense(
    256,
    activation="relu"
))

model.add(Dropout(0.30))

# ----------------------------------------------------------

model.add(Dense(
    128,
    activation="relu"
))

# ----------------------------------------------------------

model.add(Dense(
    num_classes,
    activation="softmax"
))

# ==========================================================
# COMPILE
# ==========================================================

optimizer = Adam(
    learning_rate=0.0005
)

model.compile(

    optimizer=optimizer,

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=20,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1

)

checkpoint = ModelCheckpoint(

    "best_lstm_video.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# ==========================================================
# TRAIN
# ==========================================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=150,

    batch_size=32,

    callbacks=[

        early_stop,

        reduce_lr,

        checkpoint

    ]

)

# ==========================================================
# EVALUATE
# ==========================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1

)

print("\nFinal Accuracy :", accuracy)

# ==========================================================
# SAVE
# ==========================================================

model.save("lstm_video_model.keras")

np.save(
    "label_classes_video.npy",
    encoder.classes_
)

print("\nModel Saved Successfully!")

# ==========================================================
# REPORT
# ==========================================================

predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis=1)

true = np.argmax(y_test, axis=1)

print("\nClassification Report\n")

print(

    classification_report(

        true,

        predictions,

        target_names=encoder.classes_

    )

)

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        true,

        predictions

    )

)

print("\nDone.")
