import pygame
import math
from conf import *
import os

class Car:
    def __init__(self, screen, start_dir, start_x, start_y):
        self.start_x = start_x
        self.start_y = start_y
        self.x = self.start_x
        self.y = self.start_y
        self.dir = start_dir
        self.screen = screen
        self.speed = 0
        self.wheel = 0  # -1 to 1
        self.image = pygame.image.load(os.path.join("images", "car.png"))
        self.img = None
        self.img_mask = None
        self.img_rect = None

    def reset(self):
        self.dir = 0
        self.x = self.start_x
        self.y = self.start_y

    def update_pos(self):
        self.dir -= self.wheel * self.speed
        self.x += math.cos(math.radians(self.dir)) * self.speed
        self.y -= math.sin(math.radians(self.dir)) * self.speed

    def blit(self):
        self.update_pos()
        self.img = rot_center(self.image, self.dir)
        self.img_mask = pygame.mask.from_surface(self.img)
        self.img_rect = self.img.get_rect()
        self.screen.blit(self.img, (self.x, self.y))

class Info:
    def __init__(self, screen, car, sensors):
        self.screen = screen
        self.car = car
        self.font = pygame.font.SysFont("monospace", 18)
        self.sensors = sensors

    def blit(self):
        speed_label = self.font.render("Speed: " + str(self.car.speed), 1, (255, 255, 255))
        wheel_label = self.font.render("Wheel: " + str(self.car.wheel), 1, (255, 255, 255))
        direction_label = self.font.render("Direction: " + str(self.car.dir), 1, (255, 255, 255))
        self.screen.blit(speed_label, (SCREEN_WIDTH - 180, 20))
        self.screen.blit(wheel_label, (SCREEN_WIDTH - 180, 40))
        self.screen.blit(direction_label, (SCREEN_WIDTH - 180, 60))

        for i, sensor in enumerate(self.sensors):
            label_txt = "Sensor{0}: {1}".format(i, sensor.measurement)
            label = self.font.render(label_txt, 1, WHITE)
            self.screen.blit(label, (180, i*20))


class Sensor:
    def __init__(self, screen, deg, car, circuit):
        self.screen = screen
        self.deg = deg
        self.car = car
        self.circuit = circuit
        self.measurement = None
        self.deg_x = None
        self.deg_y = None
        self.car_center_x = None
        self.car_center_y = None


    def measure(self):
        self.car_center_x = self.car.x + self.car.img_rect.center[0]
        self.car_center_y = self.car.y + self.car.img_rect.center[1]

        len = 0
        for i in range(1500):
            self.deg_x = self.car_center_x + math.cos(math.radians(self.deg + self.car.dir)) * len
            self.deg_y = self.car_center_y - math.sin(math.radians(self.deg + self.car.dir)) * len

            line_surface = pygame.Surface((2, 2), pygame.SRCALPHA)
            line_rect = line_surface.get_rect()
            line_rect.topleft = self.deg_x, self.deg_y
            line_surface.fill((255,0,0))
            if self.circuit.img_mask.overlap(pygame.mask.from_surface(line_surface),
                                             (int(line_rect[0]), int(line_rect[1]))) is not None:
                self.measurement = len
                break
            len += 1

        pass

    def blit(self):
        area_clear = 200
        if self.measurement > area_clear:
            red = 0
            green = 255
        else:
            ratio = self.measurement / area_clear
            value = 510 * ratio
            if value <= 255:
                red = 255
                green = value
            else:
                red = 255 - (value - 255)
                green = 255

        pygame.draw.line(self.screen, (int(red), int(green), 0), (self.car_center_x, self.car_center_y), (self.deg_x, self.deg_y), 2)

class Sensors:
    def __init__(self, screen, car, circuit):
        self.screen = screen
        self.car = car
        self.circuit = circuit
        self.nr_of_sensors = 7
        self.density = 20
        self.sensors = self.init_sensors()

    def init_sensors(self):
        sensors = []
        angle_range = (self.nr_of_sensors - 1) * self.density
        start_angle = -angle_range // 2

        for deg in range(start_angle + angle_range, start_angle - 1, -self.density):
            sensors.append(Sensor(self.screen, deg, self.car, self.circuit))
        return sensors

    def blit(self):
        for sensor in self.sensors:
            sensor.measure()
            sensor.blit()


class Circuit:
    def __init__(self, screen):
        self.screen = screen
        self.img = pygame.image.load(os.path.join("images", "circuit1.png"))
        self.img_mask = None
        self.img_rect = None

    def blit(self):
        self.img_mask = pygame.mask.from_surface(self.img)
        self.img_rect = self.img.get_rect()
        self.screen.blit(self.img, (0, 0))


# Rotate an image while keeping its center and size
def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image
