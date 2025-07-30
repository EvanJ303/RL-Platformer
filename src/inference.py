from itertools import count
from agent import DQNAgent
import environment

with open('./data/latest_checkpoint.txt', 'r') as f:
    checkpoint_path = f.read().strip()

agent = DQNAgent(6, 3)
agent.load(checkpoint_path)
agent.disable_exploration()

NUM_EPISODES = 3

for episode in range(NUM_EPISODES):
    state, info = environment.reset()
    episode_reward = 0.0

    for step in count():
        action = agent.select_action(state)
        next_state, reward, done, info = environment.step(action)

        episode_reward += reward
        
        state = next_state

        if done:
            print(f'Episode {episode + 1} finished. Total reward: {episode_reward}')
            break