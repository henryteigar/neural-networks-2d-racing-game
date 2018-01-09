import gym
import numpy as np
from PIL import Image
import random
from keras.layers import Conv2D, Dense, Input, Activation


H = 200
batch_size = 10
learning_rate = 1e-4
decay_rate = 0.99
gamma = 0.99
resume = True
render = False
capture_real_image = True



def preprocess_image(image):
    image = image[:-19,15:-15,1]

    image = np.delete(image, np.s_[::3], 0)
    image = np.delete(image, np.s_[::4], 1)
    image[image > 200] = 255
    image[(image > 100) & (image < 115)] = 150

    if not capture_real_image:
        image = image.astype(np.float).ravel()
    return image


def save_real_image(image):
    img_temp = []
    for i, rida in enumerate(image):
        img_temp.append([])
        for j, veerg in enumerate(rida):
            img_temp[i].append([veerg, veerg, veerg])

    img = Image.fromarray(np.array(img_temp), 'RGB')
    img.save("test.png")

def apply_NN(input):
    x = Input(shape=(2,2))
    c1 = Conv2D(16, 8, strides=4, activation="relu")(x)
    c2 = Conv2D(32, 4, strides=2, activation="relu")(c1)
    h = Dense(256)(c2)
    a1 = Activation('softmax')(h)
    a2 = Activation('relu')(h)


D = 45
#lisa salvestamine



env = gym.make('CarRacing-v0')
initial_input = env.reset()
prev_x = None

running_reward = 0
reward_sum = 0
episode_nr = 0

while True:

    if render:
        env.render()

    current_image = preprocess_image(initial_input)
    print(current_image.shape)
    x = current_image - prev_x if prev_x is not None else np.zeros(D)
    prev_x = current_image




