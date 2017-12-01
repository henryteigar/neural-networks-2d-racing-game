import pygame
from objects import *
from conf import *

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(CAPTION)
clock = pygame.time.Clock()
game_ended = False


car = Car(screen, 0, SCREEN_WIDTH / 2 - CAR_WIDTH / 2, SCREEN_HEIGHT - 80)
circuit = Circuit(screen)
sensors = Sensors(screen, car, circuit)
info = Info(screen, car, sensors.sensors)


def detect_collision():
    if circuit.img_mask.overlap(car.img_mask, (int(car.x), int(car.y))) is not None:
        car.reset()
        info.crashes += 1

speed = 3
car.speed = speed

while not game_ended:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_ended = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                car.speed = speed
            if event.key == pygame.K_a:
                car.wheel += -1
            if event.key == pygame.K_d:
                car.wheel += 1
            if event.key == pygame.K_r:
                info.crashes = 0
                car.reset()
            if event.key == pygame.K_1:
                speed = 1
                car.speed = speed
            if event.key == pygame.K_2:
                speed = 2
                car.speed = speed
            if event.key == pygame.K_3:
                speed = 3
                car.speed = speed
            if event.key == pygame.K_4:
                speed = 4
                car.speed = speed
            if event.key == pygame.K_5:
                speed = 5
                car.speed = speed
            if event.key == pygame.K_6:
                speed = 6
                car.speed = speed
            if event.key == pygame.K_0:
                speed = 0
                car.speed = speed

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                car.speed = 0
            if event.key == pygame.K_a:
                car.wheel -= -1
            if event.key == pygame.K_d:
                car.wheel -= 1

    screen.fill(WHITE)
    circuit.blit()
    car.blit()
    info.blit()
    sensors.blit()
    detect_collision()

    pygame.display.update()
    clock.tick(60)
