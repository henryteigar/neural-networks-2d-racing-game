import gym
import numpy as np
from PIL import Image
import random


env = gym.make('CarRacing-v0')
env.reset()



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



while True:
    steering = -1 + random.random()*2
    throttle = random.random()
    env.render()
    image, reward, done, info = env.step([0,throttle,0])
    image = preprocess_image(image)


    if capture_real_image:
        save_real_image(image)
