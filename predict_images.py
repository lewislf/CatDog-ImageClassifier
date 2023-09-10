import cv2 
import numpy as np
import os
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = '/path/to/your/train/train/directory'
TEST_DIR = '/path/to/your/test/test/directory'
IMG_SIZE = 50
IMAGE_PER_PAGE = 12
LR = 1e-3

MODEL_NAME = 'cats&dogs-{}-{}.model'.format(LR, '5conv-basic')

def label_img(img):
    word_label = img.split('.')[-3]

    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

def process_test_data():
    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)

    return testing_data

# train_data = create_train_data()
train_data = np.load('train_data.npy', allow_pickle=True)

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

# 1
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# 2
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# 3
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# 4
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# 5
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# 6
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# 2 = numero de classes (cat & dog)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')

# test_data = process_test_data()
test_data = np.load('test_data.npy', allow_pickle=True)

def display_images(test_data, page_number):
    start_idx = (page_number - 1) * IMAGE_PER_PAGE
    end_idx = page_number * IMAGE_PER_PAGE

    fig = plt.figure()

    for num, data in enumerate(test_data[start_idx:end_idx]):
        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.show()

# Example: Display images from page 1
page_number = 3
display_images(test_data, page_number)