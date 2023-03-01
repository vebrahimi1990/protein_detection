import numpy as np
import tensorflow as tf
from keras.models import Input, Model
from config import CFG

data_config = CFG['data']
patch_size = data_config['patch_size']
resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                 input_tensor=Input(shape=(patch_size, patch_size, 3)), classes=1000,
                                                 classifier_activation='softmax')

resnet_layer = Model(inputs=resnet.input, outputs=resnet.get_layer(resnet.layers[174].name).output)


def resnet_output_train(input_image):
    input_image = np.expand_dims(input_image, axis=-1)
    input_image = np.concatenate((input_image, input_image, input_image), axis=-1)

    aa = np.zeros((input_image.shape[0], 2, 2, 2048, input_image.shape[3]))
    for i in range(int(input_image.shape[0] / 100)):
        for j in range(input_image.shape[3]):
            aa[i * 100:(i + 1) * 100, :, :, :, j] = resnet_layer(input_image[i * 100:(i + 1) * 100, :, :, j, :])
    features = aa
    features = np.nan_to_num(features)
    features = features/features.max()
    return features


def resnet_output_test(input_image):
    input_image = np.expand_dims(input_image, axis=-1)
    input_image = np.concatenate((input_image, input_image, input_image), axis=-1)

    aa = np.zeros((input_image.shape[0], 2, 2, 2048, input_image.shape[3]))
    for i in range(int(input_image.shape[0] / 64)):
        for j in range(input_image.shape[3]):
            aa[i * 64:(i + 1) * 64, :, :, :, j] = resnet_layer(input_image[i * 64:(i + 1) * 64, :, :, j, :])
    features = aa
    features = np.nan_to_num(features)
    features = features/features.max()
    return features
