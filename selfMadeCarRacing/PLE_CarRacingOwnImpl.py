from game import CarRacingOwnImpl
from ple import PLE
import numpy as np
import time


render = False

def process_state(state):
    return np.array(state.values())

game = CarRacingOwnImpl()
p = PLE(game, fps=30, state_preprocessor=process_state, display_screen=render)

p.init()
i = 0



while True:
    # if the game is over
    if p.game_over():
        print(p.score())
        p.reset_game()

    obs = p.getScreenRGB()
    #action = game.actions['left']
    action = p.NOOP
    reward = p.act(action)

    i+=1
    if i % 60 == 0:
        p.saveScreen("screen_capture.png")





