import numpy as np
import pygame
import sys

WIDTH = 1200
HEIGHT = 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Platformer')

clock = pygame.time.Clock()
FPS = 60

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

agent = pygame.Rect(100, 300, 25, 25)
agent_speed = 5
agent_vel_y = 0
GRAVITY = 0.7
JUMP_POWER = -10
on_ground = 0

DEFAULT_STATE = (100, 300, 0, 200, 425, 0)

MAX_STEPS = 1000
step_count = 0

ground = pygame.Rect(0, 550, WIDTH, 50)

platforms = [
    pygame.Rect(200, 450, 200, 20),
    pygame.Rect(500, 350, 200, 20),
    pygame.Rect(100, 250, 200, 20)
]

objectives = [
    pygame.Rect(200, 425, 20, 20),
    pygame.Rect(500, 325, 20, 20),
    pygame.Rect(100, 225, 20, 20)
]

objective_index = 0

def step(agent_input):
    global agent_vel_y, on_ground, objective_index, step_count

    done = False
    step_count += 1
    if step_count >= MAX_STEPS:
        done = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if agent_input == 0:
        agent.x -= agent_speed
    if agent_input == 1:
        agent.x += agent_speed
    if agent_input == 2 and on_ground:
        agent_vel_y = JUMP_POWER
        on_ground = 0

    agent_vel_y += GRAVITY
    agent.y += agent_vel_y

    if agent.colliderect(ground):
        agent.y = ground.y - agent.height
        agent_vel_y = 0
        on_ground = 1
    else:
        on_ground = 0

    if agent.colliderect(objectives[objective_index]):
        new_index = objective_index

        while new_index == objective_index:
            new_index = np.random.randint(0, 3)
        
        objective_index = new_index

    screen.fill(WHITE)
    
    pygame.draw.rect(screen, RED, agent)
    pygame.draw.rect(screen, GREEN, ground)
    pygame.draw.rect(screen, BLUE, objectives[objective_index])

    for platform in platforms:
        pygame.draw.rect(screen, GREEN, platform)

    pygame.display.flip()
    clock.tick(FPS)

    state = (agent.x, agent.y, agent_vel_y, objectives[objective_index].x, objectives[objective_index].y, on_ground)
    
    reward = 0.0

    if agent.colliderect(objectives[objective_index]):
        reward = 10
    reward -= 0.1

    return state, reward, done, {}

def reset():
    global agent_vel_y, objective_index, on_ground, step_count

    agent.x = 100
    agent.y = 300
    agent_vel_y = 0
    objective_index = 0
    on_ground = 0

    step_count = 0

    screen.fill(WHITE)
    
    pygame.draw.rect(screen, RED, agent)
    pygame.draw.rect(screen, GREEN, ground)
    pygame.draw.rect(screen, BLUE, objectives[objective_index])

    for platform in platforms:
        pygame.draw.rect(screen, GREEN, platform)

    pygame.display.flip()

    return DEFAULT_STATE, {}