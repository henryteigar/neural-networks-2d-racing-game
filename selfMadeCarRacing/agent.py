import numpy as np
from game import CarRacingOwnImpl

game = CarRacingOwnImpl()

def get_action(x):
    print(x)
    return game.actions['right']