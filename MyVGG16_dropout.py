# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')


class MyVGG16_dropout():
    def __init__(self, inputs):
        self.inputs = inputs

    def CreateMyModel(self):
        base_model = keras.applications.VGG16(input_tensor=self.inputs, include_top=False, weights='imagenet')
        base_model.trainable = False
        # base_model.summary()
        for layer in base_model.layers[-4::]:
            layer.trainable = True
        fc1 = keras.layers.Dense(256, activation='relu', name='fc1_1')
        # drop1 = keras.layers.Dropout(0.2, name='dropout_1')
        fc2 = keras.layers.Dense(512, activation='relu', name='fc1_2')
        # , kernel_regularizer = keras.regularizers.l2(0.1)
        drop2 = keras.layers.Dropout(0.5, name='dropout1_2')
        fc3 = keras.layers.Dense(256, activation='relu', name='fc1_3')
        # drop3 = keras.layers.Dropout(0.3, name='dropout_3')
        fc4 = keras.layers.Dense(64, name='out1_score')
        softmax = keras.layers.Activation('softmax', name='out1')
        x = base_model.output
        x = keras.layers.GlobalAvgPool2D(name='block5_pool_GAP')(x)
        # 从epoch72加入dropout，之前都没有
        x = fc1(x)
        # x = drop1(x)
        x = fc2(x)
        x = drop2(x)
        x = fc3(x)
        # x = drop3(x)
        x = fc4(x)
        x = softmax(x)
        return keras.Model(self.inputs, x)

# inputs = keras.Input(shape=(256, 256, 3), name="images")
# b = MyVGG16_dropout(inputs).CreateMyModel()
# b.summary()
