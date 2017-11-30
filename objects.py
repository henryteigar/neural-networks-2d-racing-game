import pygame

from conf import *


class Car:
    def __init__(self, screen, start_dir, start_x, start_y):
        self.x = start_x
        self.y = start_y
        self.dir = start_dir
        self.screen = screen
        self.speed = 0

    def update_pos(self):
        self.y -= self.speed

    def blit(self):
        self.update_pos()
        pygame.draw.rect(self.screen, WHITE, (self.x, self.y, CAR_WIDTH, CAR_LENGTH))
