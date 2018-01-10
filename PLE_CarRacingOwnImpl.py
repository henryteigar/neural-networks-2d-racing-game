from game import CarRacingOwnImpl
from ple import PLE
import numpy as np
import time

game = CarRacingOwnImpl()
p = PLE(game, fps=30)

p.init()
i = 0

for i in range(np.random.randint(0, 20)):
    reward = p.act(p.NOOP)

while True:
    # if the game is over
    if p.game_over():
        p.reset_game()

    obs = p.getScreenRGB()
    action = game.actions['left']
    reward = p.act(action)
    print(reward)
    i+=1
    if i % 60 == 0:
        p.saveScreen("screen_capture.png")

    #time.sleep(1)


