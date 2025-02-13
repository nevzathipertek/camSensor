# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:51:49 2024

@author: nevza
"""

import os
import cv2
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


train_dataset_dir = r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\trainDataset"
test_dataset_dir = r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\testDataset"


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dataset_dir,
    target_size=(64, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dataset_dir,
    target_size=(64, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical')



model = tf.keras.models.Sequential([
    #tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=(64, 128)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    #tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy', #'sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
model = tf.keras.Sequential([
    
    tf.keras.layers.InputLayer((64, 128, 1)),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='Conv2D-1'),
    tf.keras.layers.MaxPooling2D(name='MaxPooling2D-1'),
    
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', name='Conv2D-2'),
    tf.keras.layers.MaxPooling2D(name='MaxPooling2D-2'),
    
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', name='Conv2D-3'),
    tf.keras.layers.MaxPooling2D(name='MaxPooling2D-3'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # İkili sınıflandırma için sigmoid
])

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
"""

model.summary()

"""
def remove_last_axis(generator):
    for batch in generator:
        images, labels = batch
        yield np.squeeze(images, axis=-1), labels

train_generator_mod = remove_last_axis(train_generator)
validation_generator_mod = remove_last_axis(validation_generator)

hist = model.fit(
    train_generator_mod,
    steps_per_epoch=len(train_generator),
    epochs=128,
    validation_data=validation_generator_mod,
    validation_steps=len(validation_generator)
)
"""


hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=200,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

"""
ipth = r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00000.jpg"
new_image = cv2.imread(ipth, cv2.IMREAD_GRAYSCALE)
new_image = new_image/255.
new_image = np.expand_dims(new_image, axis=0)
new_image = np.expand_dims(new_image, axis=-1)
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions)  # En yüksek olasılığa sahip sınıfı al
print(f'Tahmin edilen sınıf: {predicted_class}')  
"""


image_paths = [r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00000.jpg",
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00001.jpg",
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00002.jpg",
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00003.jpg",
    
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00004.jpg",
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00005.jpg",
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00006.jpg",
     r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00007.jpg",
    
    r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00008.jpg",
     r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\predictDataset\00009.jpg"]


for image_path in image_paths:
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_image = new_image/255.
    new_image = np.expand_dims(new_image, axis=0)
    #new_image = np.expand_dims(new_image, axis=-1)
    #new_image = new_image.reshape((1, 64, 128, 1))
    
    predictions = model.predict(new_image)

    predicted_class = np.argmax(predictions)  # En yüksek olasılığa sahip sınıfı al
    print(f'Tahmin edilen sınıf: {predicted_class}')    
    
    
    
model.save("model1.h5")



plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Accuracy - Validation Accuracy', fontsize=14, pad=10)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()


