import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

IMAGESIZE = 80
DATADIR = './datasets/test'
MODELDIR = './models'
# Path to 'Train' folder containing folders of pics for each letter (A, B, C, D, etc.)
CATEGORIES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'NOTHING', 'SPACE', 'DEL'
]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='path to testing dataset', action='store', default=DATADIR)
parser.add_argument('-m', '--model', help='path to model directory', action='store', default=MODELDIR)
parser.add_argument('-s', '--size', help='specify training image size', type=int, default=IMAGESIZE)
parser.add_argument('-p', '--predict', help='output redictions', action='store_true')
args = parser.parse_args()

class InvalidPathError(Exception):
    """Raised when path to training dataset is invalid"""
    pass

def get_data(test_dir):
    if os.path.isfile(test_dir):
        data_path = test_dir
    elif os.path.isdir(test_dir):
        # get last modified file in directory
        data_path = max([os.path.join(test_dir, file) for file in os.listdir(test_dir)], key=os.path.getctime)
        if not os.path.isfile(data_path):
            raise InvalidPathError
    else:
        raise InvalidPathError

    print(f"Reading {data_path} (May take a while)...")
    data = pd.read_csv(data_path, delimiter=',').values[1:]
    X_test = [[i[1:]] for i in data]
    y_test = [i[:1] for i in data]

    print("Shaping and shuffling data...")
    data = list(zip(X_test, y_test))
    random.shuffle(data)

    X_test, y_test = zip(*data)
    X_test = np.asarray(X_test).reshape(-1, args.size, args.size, 1)
    y_test = np.asarray(y_test).reshape(-1)

    return X_test, y_test

def get_model(model_dir):
    if os.path.isfile(model_dir):
        model_path = model_dir
    elif os.path.isdir(model_dir):
        model_path = max([os.path.join(model_dir, file) for file in os.listdir(model_dir)], key=os.path.getctime)
        if not os.path.isfile(model_path):
            raise InvalidPathError
    else:
        raise InvalidPathError

    return tf.keras.models.load_model(model_path)

X_test, y_test = get_data(args.data)
model = get_model(args.model)

if args.predict:
    predictions = model.predict([X_test])
    for i in range(len(X_test)):
        correct = int(y_test[i]) == np.argmax(predictions[i])
        print(f"{'✓' if correct else 'X'} {i}\tValue: {CATEGORIES[int(y_test[i])]}\tPrediction: {CATEGORIES[np.argmax(predictions[i])]}")

val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f'val_loss: {val_loss}\tval_accuracy: {val_accuracy}')

