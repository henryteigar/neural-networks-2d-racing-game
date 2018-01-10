""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym
import os.path
from keras.layers import Dense, Reshape
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model

learning_rate = 0.01
epsilon = 1e-5
decay_rate = 0.90
gamma = 0.99  # factor to discount reward

resume = False
render = False


def build_model():
    model = Sequential()
    model.add(Reshape((4,), input_shape=(4, 1)))
    # Miks RELU ei tööta nii hästi
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(25, activation="tanh"))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer=RMSprop(lr=learning_rate, decay=decay_rate, epsilon=epsilon), metrics=["accuracy"],
                   loss="categorical_crossentropy")
    return model


if resume and os.path.isfile('model_pole.h5'):
    model = load_model('model_pole.h5')
else:
    model = build_model()


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


env = gym.make("CartPole-v0")
observation = env.reset()
prev_x = None
observations, taken_actions, rewards = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

possible_actions = [0, 1]

# For information reasons only
last_n_scores = []
last_n_average_scores = []

while True:
    if render:
        env.render()

    x = observation

    a_probs = model.predict_on_batch(np.reshape(x, (1, 4, 1))).flatten() # Mis vahe on predict_on_batch() vs predict()
    prob = a_probs / np.sum(a_probs)

    action = np.random.choice(2, 1, p=prob)[0]

    observation, reward, done, info = env.step(possible_actions[action])

    taken_action = np.zeros([2])
    taken_action[action] = 1

    taken_actions.append(taken_action)
    observations.append(x)
    rewards.append(reward)
    reward_sum += reward

    if done:
        episode_number += 1

        taken_actions = np.vstack(taken_actions)
        rewards = np.vstack(rewards)
        rewards = discount_rewards(rewards)

        # ?????
        advantage = rewards - np.mean(rewards)

        X = np.reshape(observations, (len(observations), 4, 1))
        Y = taken_actions

        model.train_on_batch(X, Y, sample_weight=advantage.flatten())

        observations, taken_actions, rewards = [], [], []  # reset array memory

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
        observation = env.reset()
        prev_x = None
