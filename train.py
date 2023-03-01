import numpy as np
import tensorflow as tf
from keras.models import Input
from tensorflow import keras
from bag_generator import data_generator_train
from config import CFG
from model import fccn
from obtain_vectors_resnet import resnet_output_train

data_config = CFG['data']
model_config = CFG['model']


x_train, label_train = data_generator_train(data_config)
features = resnet_output_train(x_train)
labels = label_train[0:len(features)]

batch_size = model_config['batch_size']
ratio = data_config['ratio']
m = np.floor(ratio * features.shape[0]).astype(int)
train_tf_dataset = tf.data.Dataset.from_tensor_slices((features[0:m], labels[0:m]))
train_dataset = train_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

valid_tf_dataset = tf.data.Dataset.from_tensor_slices((features[m::], labels[m::]))
valid_dataset = valid_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

model_input = Input(features.shape[1:-1])
model_fccn = fccn(model_input, dropout=model_config['dropout'])

loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
accuracy = tf.keras.metrics.BinaryAccuracy()
optimizer = keras.optimizers.SGD(learning_rate=model_config['lr'])
model_fccn.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


epochs = model_config['n_epochs']
valid_loss = np.zeros((epochs, 1))
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = []
            for i in range(features.shape[-1]):
                logits.append(model_fccn(x_batch_train[:, :, :, :, i], training=True))
            logits = tf.math.reduce_max(logits, axis=0)
            loss_value = loss(y_batch_train, logits)

        grads = tape.gradient(loss_value, model_fccn.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_fccn.trainable_weights))

    valid_loss_value = []
    valid_accuracy = []
    for v_step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
        valid_logits = []
        for i in range(features.shape[-1]):
            valid_logits.append(model_fccn(x_batch_valid[:, :, :, :, i], training=False))
        valid_logits = tf.math.reduce_max(valid_logits, axis=0)
        valid_accuracy.append(accuracy(y_batch_valid, valid_logits))
        valid_loss_value.append(loss(y_batch_valid, valid_logits))
    valid_loss_value = tf.reduce_mean(valid_loss_value)
    valid_accuracy = tf.reduce_mean(valid_accuracy)
    valid_loss[epoch] = valid_loss_value.numpy()

    print('Validation accuracy is %.2f' % valid_accuracy)
    if epoch == 0:
        model_fccn.save_weights(model_config['save_dr'])
        print('Model is saved.')
    else:
        if valid_loss[epoch] == np.min(valid_loss[valid_loss > 0]):
            model_fccn.save_weights(model_config['save_dr'])
            print('Model is saved.')
