import os
import fnmatch
import cv2
import numpy as np
import string
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
from tensorflow.keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import string
from model import create_model

char_list = string.ascii_letters + string.digits

def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
    return dig_lst

path = r'D:\Download\90kDICT32px'

training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

i = 1
flag = 0

for root, dirnames, filenames in os.walk(path):

    for f_name in fnmatch.filter(filenames, '*.jpg'):

        img_path = os.path.join(root, f_name)
        print(f'Processing {img_path}...')
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
        else:
            print(f'File does not exist: {img_path}')

        if img is None:
            print(f'Error loading image: {img_path}. The file may not exist or is not a valid image.')
            continue
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        if w > 128 or h > 32:
            continue
        if h < 32:
            add_zeros = np.ones((32 - h, w)) * 255
            img = np.concatenate((img, add_zeros))

        if w < 128:
            add_zeros = np.ones((32, 128 - w)) * 255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img, axis=2)

        img = img / 255.

        txt = f_name.split('_')[1]

        if len(txt) > max_label_len:
            max_label_len = len(txt)

        if i % 10 == 0:
            valid_orig_txt.append(txt)
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt))

        if i == 200000:
            flag = 1
            break
        i += 1
    if flag == 1:
        break

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

model, act_model = create_model()

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "best_model5.keras"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)
batch_size = 256
epochs = 20
model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)),
          batch_size=batch_size, epochs=epochs, validation_data=(
    [valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose=1,
          callbacks=callbacks_list)

