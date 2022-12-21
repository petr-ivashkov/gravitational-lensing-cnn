from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.image import rot90
from keras.regularizers import l2
from keras.backend import maximum

import numpy as np

class RotationalConv2D(layers.Conv2D):
    def call(self, inputs):
        r0 = self.convolution_op(
            rot90(inputs, k=0) , self.kernel)   # 0° rotation
        r90 = self.convolution_op(
            rot90(inputs, k=1) , self.kernel)   # 90° rotation
        r180 = self.convolution_op(
            rot90(inputs, k=2) , self.kernel)   # 180° rotation
        r270 = self.convolution_op(
            rot90(inputs, k=3) , self.kernel)   # 270° rotation

        # result := maximum output within rotation group
        result = maximum(maximum(r0,r90),maximum(r180,r270))
        
        if self.use_bias:
            result = result + self.bias
        return result

def init(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            RotationalConv2D(filters=32, kernel_size=(11,11), activation="relu", kernel_regularizer=l2(0.001)), 
            #layers.Conv2D(filters=32, kernel_size=(11,11), activation="relu", kernel_regularizer=l2(0.001)),
            # current shape: (62,62), parameters = (11*11 + 1)*32*3 = 11712
            layers.MaxPooling2D(pool_size=(2,2), strides=2),
            # current shape: (31,31)
            layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_regularizer=l2(0.001)),
            # current shape: (25,25), parameters = (7*7*32 + 1)*64 = 100416
            layers.MaxPooling2D(pool_size=(2,2), strides=2),
            # current shape: (12,12)
            layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_regularizer=l2(0.001)),
            # current shape: (10,10), parameters = (3*3*64 + 1)*128 = 73856
            layers.Flatten(),
            # current shape: (1,1280)
            layers.Dropout(0.5),
            layers.Dense(2048, activation="relu"),
            # parameters = (10*10*128 + 1)*2048 = 26216448
            layers.Dense(512, activation="relu"),
            # parameters = (2048 + 1)*512 = 1049088
            layers.Dense(32, activation="relu"),
            # parameters = (512 + 1)*32 = 16416
            layers.Dense(1, activation="sigmoid")
            # parameters = (32 + 1)*num_classes
        ])
    return model

def train(model, x_train, y_train, batch_size, epochs, initial_epoch=0):
    # network parameters 
    m1 = 'acc'
    m2 = metrics.FalsePositives()
    m3 = metrics.FalseNegatives() 
    loss="binary_crossentropy"    
    validation_split=0.2
    learning_rate=0.0006
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    # compile and fit
    model.compile(loss=loss, optimizer=optimizer, metrics=[m1, m2, m3])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, validation_split=validation_split, shuffle=True, callbacks=[callback])
    return

def evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return

def predict_rot(model, x_sample):
    r1 = model.predict(x_sample)                            # 0°
    r2 = model.predict(np.rot90(x_sample, k=1, axes=(1,2))) # 90°
    r3 = model.predict(np.rot90(x_sample, k=2, axes=(1,2))) # 180°
    r4 = model.predict(np.rot90(x_sample, k=3, axes=(1,2))) # 270°
    r = np.array([r1,r2,r3,r4])
    return np.average(r, axis=0), np.std(r, axis=0)
