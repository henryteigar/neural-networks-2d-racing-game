""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import os
import time
import keras
from keras.layers import Conv2D, Dense, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop


# hyperparameters
#H = 200  # number of hidden layer neurons
batch_size = 1  # every how many episodes to do a param update?
learning_rate = 1e-4
epsilon = 1e-5
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
gamma = 0.99  # discount factor for reward
resume = True  # resume from previous checkpoint?
render = True

# model initialization
D = 51*49  # input dimensionality: 80x80 grid
if resume and os.path.getsize('model.p') > 0:
    print("yee")
    model = pickle.load(open('model.p', 'rb'))
#else:
    #model = {}
    #model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    #model['W2'] = np.random.randn(H) / np.sqrt(H)

#grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
#rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


#def sigmoid(x):
#    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    image = image[:-19, 15:-15, 1]

    image = np.delete(image, np.s_[::3], 0)
    image = np.delete(image, np.s_[::4], 1)
    image[image > 200] = 255
    image[(image > 100) & (image < 115)] = 150
    #image = image.astype(np.float).ravel()
    #print(image.shape)
    return image


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


#def policy_forward(x):
#    h = np.dot(model['W1'], x)
#    h[h < 0] = 0  # ReLU nonlinearity
 #   logp = np.dot(model['W2'], h)
 #   p = sigmoid(logp)
 #   return p, h  # return probability of taking action 2, and hidden state


def build_model():
    model1 = Sequential()
   # model1.add(Reshape((51, 49), input_shape=(51, 49)))
    model1.add(Conv2D(16, 19, strides=2, padding="valid", activation="relu", input_shape=(51, 49, 1)))
    model1.add(Flatten())
    model1.add(Dense(4, activation="softmax"))
    model1.compile(optimizer=RMSprop(lr=learning_rate, decay=decay_rate, epsilon=epsilon), metrics=["accuracy"],
                  loss="categorical_crossentropy")
    return model1


#def policy_backward(eph, epdlogp):
 #   """ backward pass. (eph is array of intermediate hidden states) """
 #   dW2 = np.dot(eph.T, epdlogp).ravel()
  #  dh = np.outer(epdlogp, model['W2'])
 #   dh[eph <= 0] = 0  # backpro prelu
 #   dW1 = np.dot(dh.T, epx)
 #   return {'W1': dW1, 'W2': dW2}


# env = gym.make("Pong-v0")
env = gym.make("CarRacing-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
states, gradients, actions, rewards, probs = [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

possible_actions = [[0.5, 0, 0], [-0.5, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]

maximum_reward_sum = 0
model = build_model()


while True:

    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    #print(np.array(x).shape)
    aprob = model.predict(np.reshape(x, (1, 51, 49, 1)), batch_size=batch_size).flatten()
    probs.append(aprob)

    prob = aprob / np.sum(aprob)

    action = np.random.choice(4, 1, p=prob)[0]

    #action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
    #action_vec = [0.5, 0.5, 0] if action == 2 else [-0.5, 0.5, 0]  # roll the dice!

    # record various intermediates (needed later for backprop)
    #xs.append(x)  # observation
    #hs.append(h)  # hidden state
    #y = 1 if action == 2 else 0  # a "fake label"
    #dlogps.append(
    #    y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    #
    # step the environment and get new measurements
    state, reward, done, info = env.step(possible_actions[action])

    reward_sum += reward

    actions = np.zeros([4])
    actions[action] = 1
    states.append(x)  # observation
    gradients.append(actions.astype("float32") - 1)
    rewards.append(np.clip(reward, -1, 1))  # record reward (has to be done after we call step() to get reward for previous action)


    #if reward_sum > maximum_reward_sum:
    #    maximum_reward_sum = reward_sum
    #
    #if maximum_reward_sum - reward_sum > 5:
    #    done = True


    if done:  # an episode finished
        episode_number += 1
        maximum_reward_sum = 0

        # stack together data

        gradients = np.vstack(gradients)
        epdlogp = np.vstack(actions)
        rewards = np.vstack(rewards)
        rewards = discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack(states))
        Y = probs + learning_rate * np.squeeze(np.vstack([gradients]))

        model.train_on_batch(X, Y)


        states, gradients, probs, rewards = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        #discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        #discounted_epr -= np.mean(discounted_epr)
        #discounted_epr /= np.std(discounted_epr)



        print("episode: %d - Score: %f." % (episode_number, reward_sum))


        #print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))


        if episode_number % 10 == 0: pickle.dump(model, open('model.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:
        pass
        #print('ep: {0}: game finished, reward: {1}'.format(episode_number, reward))