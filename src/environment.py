import numpy as np
import pygame
import sys

WIDTH = 1200
HEIGHT = 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Platformer')

clock = pygame.time.Clock()
FPS = 240

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

agent = pygame.Rect(588, 300, 25, 25)
agent_speed = 5
agent_vel_y = 0
GRAVITY = 0.5
JUMP_POWER = -20
on_ground = 0

DEFAULT_STATE = (588, 300, 0, 570, 305, 0)

MAX_STEPS = 3000
step_count = 0

ground = pygame.Rect(0, 550, WIDTH, 50)

platforms = [
    pygame.Rect(250, 450, 200, 20),
    pygame.Rect(500, 350, 200, 20),
    pygame.Rect(750, 450, 200, 20)
]

objectives = [
    pygame.Rect(320, 405, 60, 60),
    pygame.Rect(570, 305, 60, 60),
    pygame.Rect(820, 405, 60, 60)
]

objective_index = 1

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
    
    prev_dist = np.sqrt((agent.x - objectives[objective_index].x) ** 2 + (agent.y - objectives[objective_index].y) ** 2)

    if agent_input == 0:
        agent.x -= agent_speed
    if agent_input == 1:
        agent.x += agent_speed

    agent.x = max(0, min(WIDTH - agent.width, agent.x))
    agent.y = max(0, min(HEIGHT - agent.height, agent.y))

    if agent_input == 2 and on_ground:
        agent_vel_y = JUMP_POWER
        on_ground = 0

    agent_vel_y += GRAVITY
    agent.y += agent_vel_y

    on_ground = 0

    if agent.colliderect(ground):
        agent.y = ground.y - agent.height
        agent_vel_y = 0
        on_ground = 1
    
    for platform in platforms:
        if agent.colliderect(platform):
            if agent_vel_y > 0:
                agent.y = platform.y - agent.height
                agent_vel_y = 0
                on_ground = 1
            elif agent_vel_y < 0:
                agent.y = platform.y + platform.height
                agent_vel_y = 0

    touched_objective = agent.colliderect(objectives[objective_index])

    if touched_objective:
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

    if touched_objective:
        reward = 20.0

    curr_dist = np.sqrt((agent.x - objectives[objective_index].x) ** 2 + (agent.y - objectives[objective_index].y) ** 2)

    dist_reward = (prev_dist - curr_dist)
    reward += dist_reward

    reward -= 0.01

    under_platform = None

    for platform in platforms:
        if agent.y + agent.height > platform.y and agent.x + agent.width > platform.x - 15 and agent.x < platform.x + platform.width + 15:
            if agent.x  + agent.width / 2 > platform.x + platform.width / 2:
                under_platform = 'right'
            else:
                under_platform = 'left'
            break

    return state, reward, done, under_platform

def reset():
    global agent_vel_y, objective_index, on_ground, step_count

    agent.x = 588
    agent.y = 300
    agent_vel_y = 0
    objective_index = 1
    on_ground = 0

    step_count = 0

    screen.fill(WHITE)
    
    pygame.draw.rect(screen, RED, agent)
    pygame.draw.rect(screen, GREEN, ground)
    pygame.draw.rect(screen, BLUE, objectives[objective_index])

    for platform in platforms:
        pygame.draw.rect(screen, GREEN, platform)

    pygame.display.flip()

    return DEFAULT_STATE, False