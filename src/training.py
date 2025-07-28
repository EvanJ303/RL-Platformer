from itertools import count
from datetime import datetime
from agent import DQNAgent
import environment

NUM_EPISODES = 1000
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for episode in range(NUM_EPISODES):
    state, info = environment.reset()
    episode_reward = 0.0

    for step in count():
        action = DQNAgent.select_action(state)
        next_state, reward, done, info = environment.step(action)

        episode_reward += reward

        if done:
            next_state = None

        DQNAgent.store_experience(state, action, next_state, reward)
        
        state = next_state

        DQNAgent.optimize_model()

        if done:
            print(f'Episode {episode + 1} finished. Total reward: {episode_reward}')
            break

DQNAgent.save_model(f'models/dqn_agent_{timestamp}.pth')