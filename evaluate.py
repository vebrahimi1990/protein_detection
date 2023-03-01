import numpy as np
from keras.models import Input, Model
from bag_generator import data_generator_test
from config import CFG
from model import fccn
from obtain_vectors_resnet import resnet_output_train

data_config = CFG['data_test']
model_config = CFG['model']

x_test = data_generator_test(data_config)
features = resnet_output_train(x_test)

model_input = Input((features.shape[1:-1]))
model_fccn = fccn(model_input)
model_fccn.load_weights(model_config['save_dr'])
fccn_layer = Model(inputs=model_fccn.input, outputs=model_fccn.get_layer(model_fccn.layers[10].name).output)

prediction = np.zeros((features.shape[0], features.shape[-1]))
for i in range(len(prediction)):
    for j in range(prediction.shape[-1]):
        prediction[i:i + 1, j] = model_fccn(features[i:i + 1, :, :, :, j], training=False)

mask = np.linspace(0, len(features) - 1, len(features)).astype(int)
mask = np.unravel_index(mask, (64, 64))
c1 = mask[0]
c2 = mask[1]

for i in range(len(prediction)):
    if c2[i] + 1 < 65:
        mask[c1[i], c2[i]:c2[i] + 1] = prediction[i, :]
    else:
        mask[c1[i], c2[i]:51] = prediction[i, 51 - c2[i]:1]
