# Author: Porter Zach
# Python 3.9

import pygame
from pygame.locals import K_SPACE
import numpy as np
import random
import math
from color import hsv_to_rgb

# A particle consists of a numpy array:
# [x, y, Vx, Vy, class_index]

# A force curve consists of a numpy array:
# [min_force_distance, max_force, max_force_distance]

min_influence_min = 3
min_influence_max = 30
max_influence_min = 30
max_influence_max = 60

max_force = 10

max_nuclear_force = 50

num_particles = 100

max_init_velocity = 10
friction = .5

num_classes = 5

circle_radius = 4

screen_width = 500
screen_height = 500

def float_rand(min, max):
    return random.random() * (max - min) + min

def generate_force_curve():
    return np.array([float_rand(min_influence_min, min_influence_max), float_rand(-max_force, max_force), float_rand(max_influence_min, max_influence_max)])

def generate_color():
    return tuple([i * 255 for i in hsv_to_rgb(random.random() * 360, 1, 1)])

def reverse_lerp(x1, x2, x):
    return (x - x1) / (x2 - x1)

def triangle_value(x1, x2, x):
    mid = (x2 - x1) / 2 + x1
    if x <= mid:
        return reverse_lerp(x1, mid, x)
    else:
        return 1 - reverse_lerp(mid, x2, x)

p_positions_x = np.random.randint(screen_width, size=(num_particles, 1))
p_positions_y = np.random.randint(screen_height, size=(num_particles, 1))
p_velocities = np.random.random_sample((num_particles, 2)) * (max_init_velocity * 2) - max_init_velocity
p_classes = np.random.randint(num_classes, size=(num_particles, 1))

class_truths = [~(p_classes == i) for i in range(num_classes)]

particles = np.concatenate((p_positions_x, p_positions_y, p_velocities, p_classes), 1)

colors = [generate_color() for i in range(num_classes)]
curves = [[generate_force_curve() for i in range(num_classes)] for j in range(num_classes)]

accelerations = np.zeros((num_particles, 2))

fps = 60

pygame.display.init()
clock = pygame.time.Clock()

display = pygame.display.set_mode((screen_width, screen_height))

# the force curve of particles[p1] affects particles[p2]
def apply_force(p1, p2, curve):
    delta = particles[p2, 0:2] - particles[p1, 0:2]
    dist = np.linalg.norm(delta)
    if dist >= curve[2]: return 0
    norm = delta / dist

    if dist < curve[0]:
        accelerations[p2, 0:2] += norm * -math.log(dist / curve[0])
    else:
        accelerations[p2, 0:2] += norm * triangle_value(curve[0], curve[2], dist) * curve[1]

active = True
while active:
    display.fill((0,0,0))

    for i in range(num_particles):
        cl = int(particles[i, 4])
        for j in range(num_particles):
            if (not i == j):
                apply_force(i, j, curves[cl][int(particles[j, 4])])

    # add acceleration to velocity and reset acceleration
    particles[:, 2:4] += accelerations
    accelerations *= 0

    # add velocity to position
    particles[:, 0:2] += particles[:, 2:4]

    # contain within screen bounds
    particles[:, 0] += (particles[:, 0] < 0) * screen_width
    particles[:, 0] -= (particles[:, 0] >= screen_width) * screen_width
    particles[:, 1] += (particles[:, 1] < 0) * screen_height
    particles[:, 1] -= (particles[:, 1] >= screen_height) * screen_height

    # friction
    particles[:, 2:4] *= friction

    for i in range(num_particles):
        pygame.draw.circle(display, colors[int(particles[i, 4])], tuple(particles[i, 0:2]), circle_radius)
    pygame.display.update()

    clock.tick(fps)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
         active = False

pygame.quit()



