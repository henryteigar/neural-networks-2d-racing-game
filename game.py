import pygame
from objects import *
from conf import *
from ple.games import base
import sys

#pygame.display.set_caption(CAPTION)
#clock = pygame.time.Clock()
#game_ended = False


class CarRacingOwnImpl(base.PyGameWrapper):
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):

        actions = {
            "left": pygame.K_a,
            "right": pygame.K_d
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

    def init(self):
        self.score = 0
        self.game_over_flag = False
        self.car = Car(0, SCREEN_WIDTH / 2 - CAR_WIDTH / 2, SCREEN_HEIGHT - 80)
        self.car.speed = 1
        self.circuit = Circuit()
        self.sensors = Sensors(self.car, self.circuit)

    def getScore(self):
        return self.score

    def getGameState(self):
        measurements = []
        for sensor in self.sensors.sensors:
            measurements.append(sensor.measurement)

        state = {
            "sensors": measurements
        }
        return state

    def game_over(self):
        return self.game_over_flag

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
        self.car.blit(self.screen, dt)
        self.circuit.blit(self.screen)

        for sensor in self.sensors.sensors:
            sensor.measure()

        if self.circuit.img_mask.overlap(self.car.img_mask, (int(self.car.x), int(self.car.y))) is not None:
            self.game_over_flag = True
        else:
            self.score += 1



if __name__ == "__main__":

    pygame.init()
    game = CarRacingOwnImpl()
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()

        game.step(dt)
        pygame.display.update()
