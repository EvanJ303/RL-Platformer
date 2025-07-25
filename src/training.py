from itertools import count
import agent
import environment

NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    state, info = environment.reset()

    for step in count():
        action = agent.select_action(state)
        next_state, reward, done, info = environment.step(action)