""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym
import os
import time
import keras
from keras.layers import Conv2D, Dense, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model

learning_rate = 1e-4
epsilon = 1e-5
decay_rate = 0.90
gamma = 0.99  # factor to discount reward
batch_size = 10

resume = True
render = False

version = "v0"

from keras.losses import categorical_crossentropy
import keras.backend as K

def custom_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + 0.1 * K.sum(y_pred * K.log(y_pred), axis=-1)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, 8, strides=4, padding="valid", activation="relu", input_shape=(80, 64, 1)))
    model.add(Conv2D(64, 4, strides=2, padding="valid", activation="relu"))
    model.add(Conv2D(64, 3, strides=1, padding="valid", activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(3, activation="linear"))
    model.compile(optimizer=RMSprop(lr=learning_rate), metrics=["accuracy"],
                  loss=custom_loss)
    return model

if resume and os.path.isfile('trainingData/' + version + '/racing_model.h5'):
    model = load_model('trainingData/' + version + '/racing_model.h5')
else:
    model = build_model()


def crop_image(image):
    image = image[:-16, 16:-16, 1]
    return image


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make("CarRacing-v0")

observation = env.reset()
prev_x = None
observations, taken_actions, rewards = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

possible_actions = [[0.5, 0.3, 0], [0, 0.3, 0], [-0.5, 0.3, 0]]
maximum_reward_sum = 0
running_rewards = []

while True:
    if render:
        env.render()

    x = crop_image(observation)

    # Lahuta frame vahe
    x = x - prev_x if prev_x is not None else np.zeros(80 * 64).reshape((80, 64))  # ?

    prev_x = x

    a_probs = model.predict(np.reshape(x, (1, 80, 64, 1))).flatten()
    prob = a_probs / np.sum(a_probs)

    action = np.random.choice(3, 1, p=prob)[0]

    state, reward, done, info = env.step(possible_actions[action])

    taken_action = np.zeros([3])
    taken_action[action] = 1

    taken_actions.append(taken_action)
    observations.append(x)
    rewards.append(reward)

    reward_sum += reward

    if reward_sum > maximum_reward_sum:
        maximum_reward_sum = reward_sum

    if maximum_reward_sum - reward_sum > 5:
        done = True


    if done:
        episode_number += 1
        maximum_reward_sum = 0

        if episode_number % batch_size == 0:
            taken_actions = np.vstack(taken_actions)
            rewards = np.vstack(rewards)
            rewards = discount_rewards(rewards)
            advantages = rewards - np.mean(rewards)
            X = np.reshape(observations, (len(observations), 80, 64, 1))
            Y = taken_actions
            model.train_on_batch(X, Y, sample_weight=advantages.flatten())
            observations, taken_actions, rewards = [], [], []

        running_rewards.append(reward_sum)

        if (len(running_rewards) >= 5):
            with open("trainingData/" + version + "/history_racing.txt", "a+") as data:
                data.write(str(episode_number) + ", " + str(np.average(running_rewards)) + ", " + str(time.time()) + "\n")
            running_rewards = []


        print("episode: %d - Score: %f." % (episode_number, reward_sum))



        reward_sum = 0
        if episode_number % 1 == 0:
            if resume:
                model.save('trainingData/' + version + '/racing_model.h5')

        observation = env.reset()  # reset env
        prev_x = None
