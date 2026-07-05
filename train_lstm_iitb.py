import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

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

            sample = np.load(os.path.join(label_path, file))

            X.append(sample)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset Loaded")
print("Shape:", X.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

num_classes = len(np.unique(y))

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

print("Building model...")

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(30,126)))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nFinal Accuracy:", accuracy)

# model.save("lstm_iitb_model.keras")

# np.save("label_classes_iitb.npy", encoder.classes_)
model.save("lstm_video_model.keras")

np.save(
    "label_classes_video.npy",
    encoder.classes_
)

print("\nModel Saved Successfully!")
