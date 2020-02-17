import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import random
from keras import optimizers

IMAGE_SIZE = 80

# Path to 'Test' folder containing folders of pics for each letter (A, B, C, D, etc.)
DATADIR = 'Test'
CATEGORIES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]


def get_data():
    test_data = []
    # Store all images of all categories in an single array
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(DATADIR, category) 
        for img in os.listdir(path):
            try:
                test_data.append([prepare(os.path.join(path, img)), class_num])
            except Exception as e:
                pass
    return test_data
      

def prepare(filepath):  
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return new_img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def set_features_labels(test_data, X, y):
    for features, label in test_data:
        X.append(features)
        y.append(label)
    # Make all pics uniform resolution
    X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = np.array(y)
    return X, y
    
    
# Load Your Model Here
model = tf.keras.models.load_model('sign_lang_model.h5')

test_data = get_data()
random.shuffle(test_data)

X, y = [] # X = Features, y = Labels
X, y = set_features_labels(test_data, X, y)

# Normalize the data
X = X/255.0

results = model.evaluate(X, y, batch_size=2, verbose=1)

print('Test Loss, Test Accuracy', results)