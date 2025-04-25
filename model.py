#Model training for mnist data

import numpy as np
import tensorflow as tf

from tf_keras.models import Sequential
#Sequential helps in building the neural network
from tf_keras.datasets import mnist
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
'''
Conv2D -> Applies convolutional filters like edge detectors to the image
MaxPooling2D -> Reduces the spatial size while keeping important features
Flatten-> Turns 2D data to 1D to feed into dense layers
Dense -> Fully connected layers that Learn complex patterns
'''
from tf_keras.utils import to_categorical

#Load the mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
mnist dataset has 60,000 training images and 10,000 testing images. The images are 28X28 pixels in size.
While pre-processing the data, we'll reshape it to add the channel dimension, convert the data type to float and then scale the 
pixel value from [0, 255] to [0.0, 1.0].

Why are we normalizing ? -> It makes the training faster and prevents issue from large input images.
'''

#Pre-process the data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#Building the model
#1st model
# model = Sequential([
#     #Conv2D(number of filters/kernels, kernel size (or) size of each filter, introduce non-linearity using relu, )
#     Conv2D(32, (3,3), activation='relu', input_shape = (28, 28, 1)),
#     #MaxPooling is used to reduce the size of the image. It takes the max value out of the 2 X 2 section in the imag
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     #Dense(number of neurons in the layer, activation function to neuron's output)
#     Dense(128, activation='relu'), 
#     Dense(10, activation="softmax") 
# ])

model = Sequential([
    #Input layer
    Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(), 
    MaxPooling2D((2, 2)), 

    #2nd Convolutional layer
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)), 

    #3rd Convolutional layer
    Conv2D(256, (3, 3), activation='relu'),  # 256 filters
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(512, activation='relu'), 
    Dropout(0.5), # Dropout is used here to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.5),

    #Output layer
    Dense(10, activation='softmax') #10 classes from digits 0 to 9
]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

model.save('mnist_model.h5')

