#%% IMPORTS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

#%% CONSTANTS
IMAGE_SIZE = (224, 224)  # VGG16 expects 224x224 images
NUM_CLASSES = 4           # cataract, glaucoma, diabetic retinopathy, normal
BATCH_SIZE = 32           # smaller batch size if memory is low
EPOCHS = 10

#%% DATASET PATH
data_dir = "./dataset/dataset"  # your dataset folder containing class subfolders

#%% CHECK DEPENDENCIES
# Make sure scipy is installed (required for ImageDataGenerator)
try:
    import scipy
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "scipy"])

#%% IMAGE DATA GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # automatically split 20% for validation
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # 80% of data
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # 20% of data
)

#%% LOAD PRE-TRAINED VGG16 BASE MODEL
base_model = VGG16(
    weights='imagenet',
    include_top=False,  # remove the final classifier layer
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

base_model.trainable = False  # freeze base layers

#%% BUILD CUSTOM CLASSIFIER
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

#%% COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#%% TRAIN MODEL
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

#%% PLOT TRAINING HISTORY
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% OPTIONAL: SAVE MODEL
model.save("eye_disease_vgg16.h5")
print("Model saved as eye_disease_vgg16.h5")

#%% READY FOR TEST DATA
# If you have a separate test folder, you can create a generator like this:
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     "./dataset/test",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )
# test_loss, test_acc = model.evaluate(test_generator)
# print("Test Accuracy:", test_acc)
