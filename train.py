import os
import cv2
import random
import pickle
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

IMAGESIZE = 80
DATADIR = './datasets/train'
LOGDIR = './logs'
MODELDIR = './models'
EPOCH = 10
# Path to 'Train' folder containing folders of pics for each letter (A, B, C, D, etc.)
CATEGORIES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'NOTHING', 'SPACE', 'DEL'
]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='path to training dataset', action='store', default=DATADIR)
parser.add_argument('-l', '--log', help='path to log directory', action='store', default=LOGDIR)
parser.add_argument('-m', '--model', help='path to model directory', action='store', default=MODELDIR)
parser.add_argument('-s', '--size', help='specify training image size', type=int, default=IMAGESIZE)
parser.add_argument('-e', '--epoch', help='specify number of epochs', type=int, default=EPOCH)
args = parser.parse_args()

class InvalidPathError(Exception):
    """Raised when path to training dataset is invalid"""
    pass

def get_data(train_dir):
    if os.path.isfile(train_dir):
        data_path = train_dir
    elif os.path.isdir(train_dir):
        # get last modified file in directory
        data_path = max([os.path.join(train_dir, file) for file in os.listdir(train_dir)], key=os.path.getctime)
        if not os.path.isfile(data_path):
            raise InvalidPathError
    else:
        raise InvalidPathError

    print(f"Reading {data_path} (May take a while)...")
    data = pd.read_csv(data_path, delimiter=',').values[1:]
    X_train = [[i[1:]] for i in data]
    y_train = [i[:1] for i in data]

    print("Shaping and shuffling data...")
    data = list(zip(X_train, y_train))
    random.shuffle(data)

    X_train, y_train = zip(*data)
    X_train = np.asarray(X_train).reshape(-1, args.size, args.size, 1)
    y_train = np.asarray(y_train).reshape(-1)

    return X_train, y_train

if not os.path.isdir(args.log):
    os.makedirs(args.log)
if not os.path.isdir(args.model):
    os.makedirs(args.model)

NAME = f"sign_model_cnn_64x2_{time.time()}_{args.size}px"
tensorboard = TensorBoard(log_dir=f"{os.path.join(args.log, NAME)}")

X_train, y_train = get_data(args.data)

model = Sequential()

# Layer 1
model.add(Conv2D(64, (3,3), input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(29))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=5, epochs=args.epoch, validation_split=0.1, callbacks=[tensorboard])

model.save(f"{os.path.join(args.model, NAME+'.h5')}")

