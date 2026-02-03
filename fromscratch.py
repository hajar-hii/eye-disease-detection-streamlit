#%%
# cnn_eye_disease.py
# Run: python cnn_eye_disease.py
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#%%
# --- USER CONFIG ---
DATA_DIR = "dataset_split"         # root dataset folder
IMAGE_SIZE = (128, 128)      # resize images to this
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = None           # leave None to infer from generator
MODEL_SAVE = "final_eye_cnn_from_scratch.keras"
# --------------------
#%%
import splitfolders

splitfolders.ratio("dataset", output="dataset_split", seed=42,ratio=(0.7, 0.15, 0.15))  # train/val/test
#%%
# Create generators. If you have separate val folder, use flow_from_directory for val.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    #validation_split=0.15  # used only if val folder absent
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")    # optional
test_dir = os.path.join(DATA_DIR, "test")

use_separate_val = os.path.isdir(val_dir)

if use_separate_val:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
else:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset='training',    # from validation_split
        shuffle=True
    )
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset='validation',
        shuffle=False
    )

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
#%%
# infer number of classes
NUM_CLASSES = NUM_CLASSES or train_generator.num_classes
print("Detected classes:", train_generator.class_indices)
#%%
# --- Build CNN (from scratch) ---
def build_model(input_shape=(128,128,3), num_classes=4):
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 3
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 4 (optional deeper)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    # final layer
    if num_classes == 2:
        # binary -> single output with sigmoid (but we used categorical generators, so usually num_classes>2)
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=loss,
                  metrics=['accuracy'])
    return model

model = build_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
model.summary()
#%%
# --- Callbacks ---
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
]
#%%
# --- Train ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)
#%%
import pickle

# Save history
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

#%%
# --- Save final model (already saved by ModelCheckpoint) ---
model.save("final_" + MODEL_SAVE)
#%%
# --- Evaluate on test set ---
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test accuracy: {test_acc*100:.2f}%, Test loss: {test_loss:.4f}")
#%%
# --- Confusion matrix & classification report ---
from sklearn.metrics import confusion_matrix, classification_report
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=labels))
#%%
# --- Plot training curves ---
def plot_history(h):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(h.history['loss'], label='train loss')
    plt.plot(h.history['val_loss'], label='val loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(h.history['accuracy'], label='train acc')
    plt.plot(h.history['val_accuracy'], label='val acc')
    plt.legend(); plt.title('Accuracy')
    plt.show()

plot_history(history)
#%%
# --- Single-image prediction helper ---
from tensorflow.keras.preprocessing import image

def predict_image(img_path, model, class_indices, target_size=IMAGE_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    if preds.shape[-1] == 1:
        prob = float(preds[0][0])
        label = list(class_indices.keys())[0] if prob<0.5 else list(class_indices.keys())[1]
        result= {label: prob}
    else:
        idx = np.argmax(preds[0])
        label = list(class_indices.keys())[idx]
        result= {label: float(preds[0][idx])}
    return result,img

# Example:
prediction,img_to_show = predict_image("sample.jpg", model, train_generator.class_indices)
print(prediction)
plt.imshow(img_to_show)
plt.axis('off')
plt.title(f"Predicted: {list(prediction.keys())[0]} ({list(prediction.values())[0]*100:.2f}%)")
plt.show()
# %%
#for restart
#%% restart script for single-image prediction
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os

# --- Config ---
MODEL_PATH = "final_eye_cnn_from_scratch.keras"  # your saved model
#CLASS_INDICES_PATH = "class_indices.pkl"          # optional, to save/load indices
IMAGE_SIZE = (128, 128)                           # same as training
IMG_PATH = "sample.jpg"                           # image to predict

# --- Load model ---
model = load_model(MODEL_PATH)

# --- Load class indices ---
# Option 1: if you saved them before
class_indices = {
    "cataract": 0,
    "glaucoma": 1,
    "normal": 2,
    "diabetic retinopathy": 3
}

# --- Prediction helper ---
def predict_image(img_path, model, class_indices, target_size=IMAGE_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    if preds.shape[-1] == 1:
        prob = float(preds[0][0])
        label = list(class_indices.keys())[0] if prob<0.5 else list(class_indices.keys())[1]
        result = {label: prob}
    else:
        idx = np.argmax(preds[0])
        label = list(class_indices.keys())[idx]
        result = {label: float(preds[0][idx])}
    return result, img

# --- Predict ---


# --- Show image ---
def show_prediction(img, prediction):
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {list(prediction.keys())[0]} ({list(prediction.values())[0]*100:.2f}%)")
    plt.show()

img_list=["sample.jpg","sampleD.jpeg","sampleG.jpg"]
for img_path in img_list:
    prediction, img_to_show = predict_image(img_path, model, class_indices)
    print(prediction)
    show_prediction(img_to_show, prediction)# predict 3 times for demonstration (replace with different images as needed)                           


# --- Save class indices for future use ---
with open("class_indices.pkl", "wb") as f:
    pickle.dump(class_indices, f)

# %%
