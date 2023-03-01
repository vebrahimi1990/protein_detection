import random
import numpy as np
from tifffile import imread


def shuffler(input):
    aa = np.linspace(0, len(input) - 1, len(input)).astype(int)
    random.shuffle(aa)
    output = np.empty(input.shape, dtype=np.float64)
    for i in aa:
        output[i] = input[aa[i]]
    return output


def data_generator_train(data_config):
    patch_size = data_config['patch_size']
    threshold = data_config['threshold']
    num_bag = data_config['num_bag']
    bag_size = data_config['bag_size']
    image_dr_s1 = data_config['image_dr_s1']
    image_dr_s234 = data_config['image_dr_s234']
    image_dr_s1234 = data_config['image_dr_s1234']

    f0 = imread(image_dr_s1).astype(np.float64)
    f1 = imread(image_dr_s1234).astype(np.float64)
    f2 = imread(image_dr_s234).astype(np.float64)

    f0 = f0 / (f0.max(axis=(-1, -2))).reshape((f0.shape[0], 1, 1))
    f1 = f1 / (f1.max(axis=(-1, -2))).reshape((f1.shape[0], 1, 1))
    f2 = f2 / (f2.max(axis=(-1, -2))).reshape((f2.shape[0], 1, 1))

    f0s = shuffler(f0)
    f1s = shuffler(f1)
    f2s = shuffler(f2)

    bag = np.zeros((num_bag, f0.shape[1], f0.shape[2], bag_size))
    label = np.zeros((num_bag, 1))
    instance = np.random.rand(num_bag, 1)
    label[instance < 0.3] = 1
    random.shuffle(label)
    cc1 = np.random.randint(0, f1.shape[0] - bag_size, len(label))
    cc2 = np.random.randint(0, f2.shape[0] - bag_size, len(label))
    for i, j in enumerate(label):
        if j > 0:
            bag[i] = np.moveaxis(f1s[cc1[i]:cc1[i] + bag_size], 0, -1)
            bag_ind = np.random.randint(0, bag_size, size=1)
            ind0 = np.random.randint(0, f0.shape, size=1)
            bag[i, :, :, bag_ind] = f0s[ind0]
        else:
            bag[i] = np.moveaxis(f2s[cc2[i]:cc2[i] + bag_size], 0, -1)
    # bag = bag/(bag.max(axis=(-1, -2, -3))).reshape((bag.shape[0], 1, 1, 1))

    bags = bag
    label = label.astype(np.uint8).squeeze()
    x_train = bags
    label_train = label

    print('Data set shape is:', x_train.shape)
    return x_train, label_train


def data_generator_test(data_config):
    image_dr = data_config['image_dr']
    bag_size = 1
    num_bag = data_config['num_bag']
    f0 = imread(image_dr).astype(np.float64)

    f0 = f0 / (f0.max(axis=(-1, -2))).reshape((f0.shape[0], 1, 1))

    bag = np.zeros((num_bag, f0.shape[1], f0.shape[2], bag_size))
    cc1 = np.floor(np.linspace(0, f0.shape[0] - bag_size, num_bag)).astype(int)
    for i in range(num_bag):
        bag[i] = np.moveaxis(f0[cc1[i]:cc1[i] + bag_size], 0, -1)
    # bag = bag/(bag.max(axis=(-1, -2, -3))).reshape((bag.shape[0], 1, 1, 1))

    x_test = bag
    print('Data set shape is:', bag.shape)
    return x_test
