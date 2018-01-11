import ast

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym
import os
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

learning_rate = 1e-4
gamma = 0.99  # factor to discount reward
batch_size = 10

resume = False
render = False

version = "v0"

if resume and not os.path.isdir('trainingData/' + version):
    os.makedirs('trainingData/' + version)


def build_model():
    x = Input(shape=(53, 48, 1))
    c1 = Conv2D(32, 16, strides=1, padding="valid", activation="relu")(x)
    c2 = Conv2D(64, 8, strides=1, padding="valid", activation="relu")(c1)
    c3 = Conv2D(64, 8, strides=1, padding="valid", activation="relu")(c2)
    f1 = Flatten()(c3)
    h1 = Dense(512, activation="relu")(f1)
    p = Dense(4, activation="softmax")(h1)
    model = Model(inputs=x, outputs=p)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


if resume and os.path.isfile('trainingData/' + version + '/racing_model.h5'):
    model = load_model('trainingData/' + version + '/racing_model.h5')
else:
    model = build_model()


def crop_image(image):
    image = image[:-16, 16:-16, 1]
    return image


x_train = []
y_train = []

f = open("trainingData/supervised_data.txt", "r")
read = f.readlines()[:100]
f.close()


def are_equal(l1, l2):
    if (len(l1) != len(l2)):
        return False
    for i, el in enumerate(l1):
        if l2[i] != el:
            return False
    return True

nr = 0

for rida in read:
    nr += 1

    if nr % 100 == 0:
        print(nr)

    split_indx = rida.find("], [")
    fst = rida[2:split_indx].split(", ")
    snd = rida[split_indx + 4:-3].split(", ")
    fst = [float(x) for x in fst]
    snd = [float(x) for x in snd]

    x_train.append(fst)

    action = snd

    if are_equal(action, [-0.3, 0.05, 0]):
        action = [1, 0, 0, 0]
    elif are_equal(action, [0, 0.05, 0]):
        action = [0, 1, 0, 0]
    elif are_equal(action, [0.3, 0.05, 0]):
        action = [0, 0, 1, 0]
    elif are_equal(action, [0, 0, 0.5]):
        action = [0, 0, 0, 1]
    else:
        print("Error")
    y_train.append(action)

x_train = np.reshape(x_train, (100, 53, 48, 1))

x_train -= np.mean(x_train)
x_train /= np.std(x_train)

model.fit(x_train, y_train, batch_size=10, epochs=5, validation_split=0.05)
