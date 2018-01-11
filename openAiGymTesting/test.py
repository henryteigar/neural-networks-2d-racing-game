import pygame
import gym
from random import randint
from conf import *
import numpy as np
from PIL import Image

pygame.init()
pygame.display.set_caption("TEST")
clock = pygame.time.Clock()
pygame.display.set_mode((20,10))

cur_dir = 1
breaking = False

def crop_image(image):
    image = image[:-16, 16:-16, 1]
    image = np.delete(image, np.s_[::3], 0)
    image = np.delete(image, np.s_[::4], 1)
    # image[:,:,0] = image[:,:,1]
    # image[:,:,2] = image[:,:,1]

    image[image == 204] = 255
    image[image == 229] = 255
    image[image == 102] = 125
    image[image == 103] = 125
    image[image == 104] = 125
    image[image == 105] = 125
    image[image == 106] = 125
    image[image == 107] = 125
    return image.flatten()


env = gym.make("CarRacing-v0")
observation = crop_image(env.reset())

actions = [[-1,0.1,0],[0,0.1,0],[1,0.5,0], [0,0,0.5]]





while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                cur_dir = 0
            elif event.key == pygame.K_RIGHT:
                cur_dir = 2
            elif event.key == pygame.K_DOWN:
                breaking = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                cur_dir = 1
            elif event.key == pygame.K_RIGHT:
                cur_dir = 1
            elif event.key == pygame.K_DOWN:
                breaking = False

    env.render()
    if not breaking:
        action = actions[cur_dir]
    else:
        action = actions[-1]

    observation, reward, done, info = env.step(action)
    observation = crop_image(observation)



    # w, h = 53, 48
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # img = Image.fromarray(observation, 'RGB')
    # img.save('my.png')

    pygame.display.flip()
    clock.tick(30)
