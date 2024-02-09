import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import datetime


def preprocess_image(image_path):
    
    original_image = cv2.imread(image_path)
    
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
   
    light_orange = (168, 50, 50)
    dark_orange = (182, 255, 255)
    
    mask = cv2.inRange(hsv_image, light_orange, dark_orange)
   
    segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    segmented_image = cv2.resize(segmented_image, (224, 224))
    return segmented_image


Classifier = load_model('path\Models\model.h5')


test_image_path = 'path\Test-images\Original\\abc.jpg'


segmented_image = preprocess_image(test_image_path)
segmented_image_path = os.path.join('path\Test-images\Segmented', 'segmented_' + os.path.basename(test_image_path))
cv2.imwrite(segmented_image_path, segmented_image)


Test_seg = np.expand_dims(segmented_image, axis=0)
Test_seg = Test_seg / 255.0  


start = datetime.datetime.now()
classes = Classifier.predict(Test_seg)
predicted_class_index = np.argmax(classes)
print('Predicted index:', predicted_class_index)


class_labels = ['Beginning Level', 'Early Level', 'Pre Level', 'Pro Level']
predicted_class_label = class_labels[predicted_class_index]
print('Predicted Level:', predicted_class_label)

finish = datetime.datetime.now()
elapsed = finish - start
print('________________\n')
print('Total time elapsed: ', elapsed)

