#Furkan ÇOLAK


# Import necessary libraries from TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory setup for training data
train_dir = 'train/'  # Replace with your train folder path
image_size = (48, 48)  # Size to which images will be resized
batch_size = 32  # Number of images processed before the model is updated

# Setup for preprocessing and augmentation of training images
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale images by normalizing pixel values
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,  # Resize images to 48x48
    batch_size=batch_size,  # Batch size for training
    color_mode='grayscale',  # Images are in grayscale
    class_mode='categorical'  # Assuming multiple classes in a classification task
)

# Building the convolutional neural network model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # 32 filters, 3x3 kernel, ReLU activation
    MaxPooling2D(2, 2),  # Max pooling with a 2x2 window
    Flatten(),  # Flatten the output for the dense layer
    Dense(128, activation='relu'),  # Dense layer with 128 units
    Dense(7, activation='softmax')  # Output layer with 7 units (for 7 classes), softmax for multi-class classification
])

# Compile the model with the Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(train_generator, epochs=10)  # Train for 10 epochs

# Save the trained model to a file
model.save('facial_emotion_model.h5')  # The model is saved as 'facial_emotion_model.h5'
