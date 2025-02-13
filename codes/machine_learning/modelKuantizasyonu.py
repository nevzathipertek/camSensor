# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:55:16 2024

@author: nevza
"""
import os
import cv2
import pandas as pd
import numpy as np

import tensorflow as tf


train_dataset_dir = r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\trainDataset"
test_dataset_dir = r"C:\Users\nevza\OneDrive\Desktop\AiWork\NeedleDataset\testDataset"


model = tf.keras.models.load_model('model1.h5')


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




converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_dataset():
    for _ in range(100):
        input_data = np.random.rand(1, 64, 128, 1).astype(np.float32)
        yield [input_data]


converter.representative_dataset = representative_dataset


converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8  

tflite_model_quant = converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model_quant)






#controlling
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:", input_details)
print("Output Details:", output_details)



