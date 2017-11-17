# testing model 2 and model 5 longer epoch

import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
import pickle

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils

from sklearn.utils import shuffle

print('-----------------------------------')
print('load data')
print('-----------------------------------')


f = open(r'../ships-in-satellite-imagery/shipsnet_v2.json')
dataset = json.load(f)
f.close()

df = pd.DataFrame(dataset)
new_df = df.drop(['locations','scene_ids'], axis=1)

## shuffle data
new_df = shuffle(new_df)
data = list(new_df.data)
data = np.array(data).astype('uint8')
labels = list(new_df['labels'])
labels = np.array(labels).astype('uint8')

## prepare data for CNN
x = data / 255.
x = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
y = to_categorical(labels, num_classes=2)


print('-----------------------------------')
print('TRAINING START')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 5_2')
print('-----------------------------------')


## model_5_2 train (200 epoch)
model_5_2 = Sequential()
model_5_2.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_5_2.add(MaxPooling2D(pool_size=(2, 2)))

model_5_2.add(Dropout(0.2))

model_5_2.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_5_2.add(MaxPooling2D(pool_size=(2, 2)))

model_5_2.add(Dropout(0.2))

model_5_2.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_5_2.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_5_2.add(MaxPooling2D(pool_size=(2, 2)))

model_5_2.add(Dropout(0.2))

model_5_2.add(Flatten())
model_5_2.add(Dense(512, activation='relu'))
model_5_2.add(Dropout(0.2))
model_5_2.add(Dense(2, activation='softmax'))

## compile a model
model_5_2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_5_2 = model_5_2.fit(x, y, batch_size=64, epochs=200, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 5_2')
print('-----------------------------------')

with open('../cnn_model/classifer_2/model5_2_hist.pkl', 'wb') as file_name:
    pickle.dump(history_5.history, file_name)

model_5.save('../cnn_model/classifer_2/model5_2.h5')

print('-----------------------------------')
print('end train model 5_2')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 6_2')
print('-----------------------------------')


## model_6_2 train (200 epoch)
model_6_2 = Sequential()
model_6_2.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_6_2.add(MaxPooling2D(pool_size=(2, 2)))

model_6_2.add(Dropout(0.5))

model_6_2.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_6_2.add(MaxPooling2D(pool_size=(2, 2)))

model_6_2.add(Dropout(0.5))

model_6_2.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_6_2.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_6_2.add(MaxPooling2D(pool_size=(2, 2)))

model_6.add(Dropout(0.5))

model_6_2.add(Flatten())
model_6_2.add(Dense(512, activation='relu'))
model_6_2.add(Dropout(0.5))
model_6_2.add(Dense(2, activation='softmax'))

## compile a model
model_6_2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_6_2 = model_6_2.fit(x, y, batch_size=64, epochs=200, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 6_2')
print('-----------------------------------')

with open('../cnn_model/classifer_2/model6_2_hist.pkl', 'wb') as file_name:
    pickle.dump(history_6.history, file_name)

model_6.save('../cnn_model/classifer_2/model6_2.h5')

print('-----------------------------------')
print('end train model 6_2')
print('-----------------------------------')

print('----------------------------------------------------------------')