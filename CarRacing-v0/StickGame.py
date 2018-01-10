""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import os
import time
import os.path
import keras
from keras.layers import Conv2D, Dense, Reshape, Flatten, Input
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.models import load_model



# hyperparameters
#H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 0.01
epsilon = 1e-5
decay_rate = 0.90  # decay factor for RMSProp leaky sum of grad^2
gamma = 0.99  # discount factor for reward
resume = False  # resume from previous checkpoint?
render = False

# model initialization
#D = 51*49  # input dimensionality: 80x80 grid

def build_model():
    model1 = Sequential()
    model1.add(Reshape((4,), input_shape=(4, 1)))
    model1.add(Dense(100, activation="tanh"))
    model1.add(Dense(25, activation="tanh"))
    model1.add(Dense(2, activation="softmax"))
    model1.compile(optimizer=RMSprop(lr=learning_rate, decay=decay_rate, epsilon=epsilon), metrics=["accuracy"],
                   loss="categorical_crossentropy")
    return model1


if resume and os.path.isfile('model_pole.h5'):
    model = load_model('model_pole.h5')
else:
    model = build_model()

def prepro(image):
    image = image[:-16, 16:-16, 1]
    return image


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):

        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


env = gym.make("CartPole-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
states, actions, rewards, probs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

possible_actions = [0, 1]

maximum_reward_sum = 0


last_n_scores = []
last_n_average_scores = []
while True:

    if render:
        env.render()

    x = observation

    #x = x - prev_x if prev_x is not None else np.zeros(4)
    #prev_x = x

    aprob = model.predict(np.reshape(x, (1 , 4, 1))).flatten()
    prob = aprob / np.sum(aprob)

    action = np.random.choice(2, 1, p=prob)[0]

    observation, reward, done, info = env.step(possible_actions[action])
    reward_sum += reward


    as1 = np.zeros([2])
    as1[action] = 1
    actions.append(as1)

    states.append(x)  # observation

    rewards.append(reward)



    if done:

        episode_number += 1

        maximum_reward_sum = 0

        actions = np.vstack(actions)

        rewards = np.vstack(rewards)

        rewards = discount_rewards(rewards)

        advantage = rewards - np.mean(rewards)


        X = np.array(states)
        Y = np.array(actions)







        model.train_on_batch(np.reshape(X, (len(X), 4, 1)), Y, sample_weight=advantage.flatten())
        #print("Advantages: ", np.sum(advantages))

        actions, probs, rewards = [], [], []  # reset array memory
        states = []
        if len(last_n_scores) >= 10:
            print("Average score: " + str(np.average(last_n_scores)))
            last_n_average_scores.append(np.average(last_n_scores))
            last_n_scores = []
        else:
            last_n_scores.append(reward_sum)

        if len(last_n_average_scores) >= 10:
            print("LAST TEN AVG SCORES: " + str(np.average(last_n_average_scores)))
            last_n_average_scores = []




        if episode_number % 500 == 0:
            model.save('model_pole.h5')
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

