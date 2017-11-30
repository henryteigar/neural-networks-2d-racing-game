import pygame

from objects import *
from conf import *

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(CAPTION)
clock = pygame.time.Clock()
game_ended = False

car = Car(screen, 0, SCREEN_WIDTH / 2 - CAR_WIDTH / 2, SCREEN_HEIGHT - CAR_LENGTH - 10)


while not game_ended:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_ended = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                car.speed = 3

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                car.speed = 0

    screen.fill(BLACK)

    car.blit()



    pygame.display.update()
    clock.tick(60)
