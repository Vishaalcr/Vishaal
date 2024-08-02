TITLE
Handwritten digit prediction 

OBJECTIVE
Handwritten digit prediction with machine learning

IMPORT LIBRARY

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

IMPORT DATA AND PREPROCESSING

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


TRAIN TEST SPLIT

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

     
MODELING

model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    Dense(10, activation='softmax') # Output layer with 10 units for each digit
])


     
MODEL EVALUATION

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')



     
PREDICTION

# Predict the class of the first test image
predictions = model.predict(x_test)
print(f'Predicted class for the first test image: {predictions[0].argmax()}')

EXPLANATION

1.Install Required Libraries:
Ensure you have the necessary libraries installed. You can use TensorFlow or PyTorch, but this example will use TensorFlow and Keras.
 
2.IMPORT DATA AND PREPROCESSING
The MNIST dataset contains 70,000 images of handwritten digits (0-9), split into training and test
Normalize the image data to the range [0, 1] and reshape it to fit the input requirements of the model.

3.Train test split

Fit the model to the training data and validate it on the test data.

4.Modeling
Create a Convolutional Neural Network (CNN) with layers suitable for image classification

5.Model evaluation

Check the model's performance on the test dataset.

6.Prediction

Use the trained model to make predictions on new data.
To get the class label for a single image




