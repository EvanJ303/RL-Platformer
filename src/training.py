from itertools import count
from datetime import datetime
from agent import DQNAgent
import environment


agent = DQNAgent(6, 3)
NUM_EPISODES = 100
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for episode in range(NUM_EPISODES):
    state, under_platform = environment.reset()
    episode_reward = 0.0

    for step in count():
        action = agent.select_action(state, under_platform)
        next_state, reward, done, under_platform = environment.step(action)

        episode_reward += reward

        if done:
            next_state = None

        agent.store_experience(state, action, next_state, reward)
        
        state = next_state

        agent.optimize_model()

        if done:
            print(f'Episode {episode + 1} finished. Total reward: {episode_reward}.')
            agent.decay_epsilon()
            break

checkpoint_path = f'./data/models/dqn_agent_{timestamp}.pth'

agent.save(checkpoint_path)

with open('./data/latest_checkpoint.txt', 'w') as f:
    f.write(checkpoint_path)