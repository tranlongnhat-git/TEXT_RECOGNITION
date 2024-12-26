# model.py
import string
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda, Bidirectional, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

char_list = string.ascii_letters + string.digits

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_model(input_shape=(32, 128, 1), char_list=char_list):

    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

    act_model = Model(inputs, outputs)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    return model, act_model

model, act_model = create_model()
act_model.summary()

