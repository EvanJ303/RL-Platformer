from itertools import count
from agent import DQNAgent
import environment

with open('./data/latest_checkpoint.txt', 'r') as f:
    checkpoint_path = f.read().strip()

agent = DQNAgent(6, 3)
agent.load(checkpoint_path)
agent.inference_mode()

NUM_EPISODES = 10

for episode in range(NUM_EPISODES):
    state, under_platform = environment.reset()
    episode_reward = 0.0

    for step in count():
        action = agent.select_action(state, under_platform)
        next_state, reward, done, under_platform = environment.step(action)

        episode_reward += reward
        
        state = next_state

        if done:
            print(f'Episode {episode + 1} finished. Total reward: {episode_reward}')
            break