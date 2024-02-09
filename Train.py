import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

original_data_dir = 'path\Datasets\Original'
segmented_data_dir = 'path\Datasets\Segmented'
model_path = 'model.h5'

img_width, img_height = 224, 224
batch_size = 32


train_datagen = ImageDataGenerator(rescale=1./255)

train_original_generator = train_datagen.flow_from_directory(
    original_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

def preprocess_image(image_path):
   
    original_image = cv2.imread(image_path)
   
    segmented_image = original_image  
  
    segmented_image = cv2.resize(segmented_image, (img_width, img_height))
    return segmented_image


for subdir, dirs, files in os.walk(original_data_dir):
    for file in files:
       
        original_image_path = os.path.join(subdir, file)
        segmented_image = preprocess_image(original_image_path)
        segmented_image_path = os.path.join(segmented_data_dir, os.path.relpath(subdir, original_data_dir), file)
        os.makedirs(os.path.dirname(segmented_image_path), exist_ok=True)
        cv2.imwrite(segmented_image_path, segmented_image)

train_segmented_generator = train_datagen.flow_from_directory(
    segmented_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(
    train_segmented_generator,
    steps_per_epoch=train_segmented_generator.samples // batch_size,
    epochs=10)


model.save(model_path)
