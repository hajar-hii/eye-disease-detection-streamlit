#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from tensorflow.keras.utils import image_dataset_from_directory

#%%
# Prevent TensorFlow from locking up all GPU VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#%%
training_set = image_dataset_from_directory(
    'dataset_split/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=8,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
).prefetch(buffer_size=tf.data.AUTOTUNE)
#%%
validation_set = image_dataset_from_directory(
    'dataset_split/val',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=8,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
).prefetch(buffer_size=tf.data.AUTOTUNE)
#%%
training_set
#%%
INPUT_SHAPE = (224, 224, 3)

#%%
mobnet = tf.keras.applications.MobileNetV3Large(
    input_shape=INPUT_SHAPE,
    alpha=1.0,
    minimalistic=False,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling='avg',
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
    # name="MobileNetV3Large",
)

#%%
mobnet.trainable = False

#%%
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=INPUT_SHAPE))
model.add(mobnet)
model.add(tf.keras.layers.Dense(units = 4,activation='softmax'))

#%%
metrics_list = ['accuracy',
                tf.keras.metrics.F1Score()]

#%%
# Define a scheduler to prevent the model from plateauing
initial_learning_rate = 0.0005  # Slightly higher than your current 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

#%%
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=lr_schedule),loss='categorical_crossentropy',metrics=metrics_list,jit_compile=False)
#%%
model.summary()

#%%
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=15)

#%%
model.save("Trained_Model.keras")

#%%
# Saving history
with open('Training_history.pkl', 'wb') as f:
    pickle.dump(training_history.history, f)

#%%
with open('Training_history.pkl', 'rb') as f:
    load_history = pickle.load(f)
load_history

#%%
#Loss Visualization
epochs = [i for i in range(1,16)]
plt.plot(epochs,load_history['loss'],color='red',label='Training Loss')
plt.plot(epochs,load_history['val_loss'],color='blue',label='Validation Loss')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Loss Result')
plt.legend()
plt.show()

#%%
test_set = image_dataset_from_directory(
    'dataset_split/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=8,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

#%%
test_loss,test_acc,f1_score = model.evaluate(test_set)
test_loss
test_acc
f1_score

#%%
'-----------------------MODEL EVALUATION-----------------------------'
model = tf.keras.models.load_model("Trained_Model.keras")

#%%
model.summary()

#%%
test_set = image_dataset_from_directory(
    'dataset_split/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=8,
    image_size=(224, 224),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

#%%
##Computing True labels from test set
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

#%%
Y_true

#%%
##Computing Predicted labels from test set
predicted_categories = model.predict(test_set)
Y_pred = tf.argmax(predicted_categories, axis=1)
#%%
Y_true
#%%
Y_pred

#%%
from sklearn.metrics import classification_report
print(classification_report(Y_true,Y_pred))

#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_true,Y_pred)
cm

#%%
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,annot_kws={"size":8})
plt.xlabel("Predicted Class",fontsize=10)
plt.ylabel("Actual Class",fontsize=10)
plt.title("Human Eye Disease Prediction Confusion Matrix",fontsize=12)
plt.show()
# %%
