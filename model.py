from keras.activations import sigmoid
from keras.layers import Dense, Flatten
from keras.layers import Dropout, LeakyReLU, BatchNormalization
from keras.models import Model


def fccn(inputs, dropout=0.3):
    x = inputs
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(dropout)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(dropout)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(1)(x)
    x = sigmoid(x)
    model = Model(inputs=[inputs], outputs=x)
    return model

