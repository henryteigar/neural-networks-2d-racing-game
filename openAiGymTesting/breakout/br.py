
import numpy as np
import _pickle as pickle
import gym
import os
import time
import keras
import os.path



#test2
H = 200
batch_size = 10
learning_rate = 1e-4
decay_rate = 0.99
gamma = 0.99
resume = True
render = False

# model initialization
D = 5832  # input dimensionality: 80x80 grid

if not (os.path.isfile('save.p')):
    open("save.p", "w+")


if resume and os.path.getsize('save.p') > 0:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    I = I[32:193,8:152]
    I = I[::2, ::2, 0]
    I[I != 0] = 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Breakout-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
cur_lives = 5
running_rewards = []
while True:
    env.step(1)
    if render: env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = policy_forward(x)

    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(
        y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    #if reward != 0:
    #    print(reward)
    lives = info["ale.lives"]

    if lives < cur_lives:
        reward -= cur_lives - lives
        cur_lives = lives
    elif lives > cur_lives:
        cur_lives = 5

    # if (reward != 0):
    #     print(reward)

    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1


        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        running_rewards.append(running_reward)

        if (len(running_rewards)) >= 10:
            with open("history.txt", "a+") as data:
                data.write(str(episode_number) + ", " + str(np.average(running_rewards)) + ", " + str(time.time()) + "\n")

            running_rewards = []

        if episode_number % 10 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:
        pass
        #print('ep: {0}: game finished, reward: {1}'.format(episode_number, reward))
