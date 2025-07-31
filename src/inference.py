# Itertools for iteration
from itertools import count
# Custom DQN agent
from agent import DQNAgent
# Custom environment
import environment

# Determine the latest checkpoint path using the text file
with open('./data/latest_checkpoint.txt', 'r') as f:
    checkpoint_path = f.read().strip()

# Initialize DQN agent with a state size of 6 and an action size of 3
agent = DQNAgent(6, 3)
# Load the checkpoint
agent.load(checkpoint_path)
# Set the agent to inference mode
agent.inference_mode()

# Set the number of episodes
NUM_EPISODES = 5

# Repeat for each episode
for episode in range(NUM_EPISODES):
    # Reset the environment and receive the state and under-platform state
    state, under_platform = environment.reset()
    # Set the episode reward to 0
    episode_reward = 0.0

    # Repeat for each step
    for step in count():
        # Select the action using the state and under-platform state
        action = agent.select_action(state, under_platform)
        # Input the action into the environment and receive the next state, reward, done, and under-platform state
        next_state, reward, done, under_platform = environment.step(action)

        # Update the episode reward with the reward
        episode_reward += reward
        
        # Update the state
        state = next_state

        # Check if the episode is over
        if done:
            # Print the episode and episode reward
            print(f'Episode {episode + 1} finished. Total reward: {episode_reward}')
            break