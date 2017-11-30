import pygame
import conf

pygame.init()
screen = pygame.display.set_mode((conf.width, conf.height))
pygame.display.set_caption(conf.caption)
clock = pygame.time.Clock()
game_ended = False

while not game_ended:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_ended = True

    screen.fill(conf.BLACK)
    pygame.display.update()
    clock.tick(24)
