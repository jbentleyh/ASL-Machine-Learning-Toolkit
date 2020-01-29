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

# Path to Test folder containing A, B, C etc folders
DATADIR = ''
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

IMAGE_SIZE = 80

def get_value(result):
    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'


def get_data():
    test_data = []
    #model = tf.keras.models.load_model('sign_lang_model.h5')

    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(DATADIR, category) 
        for img in os.listdir(path):
            try:
                # imag = prepare(os.path.join(path, img))
                # imag = np.array(imag).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
                # result = model.predict(imag)
                # print(get_value(result))
                test_data.append([prepare(os.path.join(path, img)), class_num])
            except Exception as e:
                pass

    return test_data
      

def prepare(filepath):  
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return new_img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
model = tf.keras.models.load_model('sign_lang_model.h5')

test_data = get_data()
random.shuffle(test_data)

X = []
y = []

for features, label in test_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
y = np.array(y)

X = X/255.0

results = model.evaluate(X, y, batch_size=2, verbose=1)
print('test loss, test acc', results)