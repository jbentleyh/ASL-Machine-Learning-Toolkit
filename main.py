import os
import cv2 # image operations
import random
import pickle # database
import numpy as np # array operations
import matplotlib.pyplot as plt # visualize data operations
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

IMAGE_SIZE = 80

# Path to Folder Containing Folders of pics (A, B, C, etc.)
DATADIR = 'sign-language-mnist\images\Train'
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
def create_training_data():
    training_data = []
    # Store all images of all categories in an single array
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    return training_data

X = [] # Features
y = [] # Labels

training_data = create_training_data()

random.shuffle(training_data)

for features, label in training_data:
    X.append(features)
    y.append(label)

# Make all pics uniform resolution
X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
y = np.array(y)

# Make and store the imformation in the database
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

model = Sequential()

# Layer 1
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
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

# Layer 4
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer ?
model.add(Flatten())
model.add(Dense(64))

# Layer ?
model.add(Dense(26))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

model.fit(X, y, batch_size=5, epochs=10, validation_split=0.1)

model.save('sign_lang_model.h5')