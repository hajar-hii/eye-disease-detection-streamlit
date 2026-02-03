# CNN for Handwritten Digit Classification (MNIST)
#%%
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
#%%
import matplotlib.pyplot as plt

#%%
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#%%
# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#%%
# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
#%%
# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#%%
# Train model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

#%%
# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

#%%
# Optional: visualize a few predictions
import numpy as np
predictions = model.predict(x_test[:5])
for i, pred in enumerate(predictions):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(pred)}")
    plt.show()


# %%
