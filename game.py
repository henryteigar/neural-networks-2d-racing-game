import pygame
from objects import *
from conf import *
from ple.games import base
import sys

#pygame.display.set_caption(CAPTION)
#clock = pygame.time.Clock()
#game_ended = False


class CarRacingOwnImpl(base.PyGameWrapper):
    def __init__(self, width=256, height=256):

        actions = {
            "left": pygame.K_a,
            "right": pygame.K_d
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        #self.fruit_size = percent_round_int(height, 0.06)
        #self.fruit_fall_speed = 0.00095 * height

        #self.player_speed = 0.021 * width
        #self.paddle_width = percent_round_int(width, 0.2)
        #self.paddle_height = percent_round_int(height, 0.04)

        #self.dx = 0.0
        #self.init_lives = init_lives

    def init(self):
        self.score = 0

        self.car = Car(0, SCREEN_WIDTH / 2 - CAR_WIDTH / 2, SCREEN_HEIGHT - 80)
        self.car.speed = 1
        self.circuit = Circuit()
        #self.sensors = Sensors(self.car, self.circuit)
        #self.info = Info(self.car, self.sensors.sensors)

    def getScore(self):
        return self.score

    def game_over(self):
        return False

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions['left']:
                    self.car.wheel -= 0.5
                if key == self.actions['right']:
                    self.car.wheel += 0.5

    def step(self, dt):
        self.screen.fill(WHITE)
        self._handle_player_events()
        self.score += self.rewards["tick"]
        self.car.update_pos()


        self.circuit.blit(self.screen)
        self.car.blit(self.screen)


#
# car = Car(screen, 0, SCREEN_WIDTH / 2 - CAR_WIDTH / 2, SCREEN_HEIGHT - 80)
# circuit = Circuit(screen)
# sensors = Sensors(screen, car, circuit)
# info = Info(screen, car, sensors.sensors)

#
# def detect_collision():
#     if circuit.img_mask.overlap(car.img_mask, (int(car.x), int(car.y))) is not None:
#         car.reset()
#         info.crashes += 1

# speed = 2
# car.speed = speed

# while not game_ended:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             game_ended = True
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_a:
#                 car.wheel += -1
#             if event.key == pygame.K_d:
#                 car.wheel += 1
#             if event.key == pygame.K_r:
#                 info.crashes = 0
#                 car.reset()
#
#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_a:
#                 car.wheel -= -1
#             if event.key == pygame.K_d:
#                 car.wheel -= 1
#
#     screen.fill(WHITE)
#     circuit.blit()
#     car.blit()
#     #info.blit()
#     #sensors.blit()
#
#     detect_collision()
#     for sensor in sensors.sensors:
#         sensor.measure()
#         print(sensor.measurement, end=" - ")
#     print()
#     pygame.display.update()
#     clock.tick_busy_loop(30)
#


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = CarRacingOwnImpl()
    #game.rng = np.random.RandomState(24)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            print("a")
            game.reset()

        game.step(dt)
        pygame.display.update()
