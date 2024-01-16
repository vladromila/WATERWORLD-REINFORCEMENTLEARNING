import matplotlib.pyplot as plt
import numpy as np

# Sample input string
input_string2 = """
Episode: 1, Total Reward: 7.0, Avg Reward per episode: 7.0
"""

# Split the input string into lines
lines = input_string2.strip().split('\n')

# Initialize lists to store total and average rewards
total_rewards = []
avg_rewards = []

# Extract total and average rewards from each line
for line in lines:
    parts = line.split(',')
    total_reward = float(parts[1].strip().split(': ')[1])
    avg_reward = float(parts[2].strip().split(': ')[1])
    total_rewards.append(total_reward)
    avg_rewards.append(avg_reward)

# Create episode numbers from 1 to the number of episodes
episode_numbers = np.arange(1, len(total_rewards) + 1)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Plot total rewards with smoother lines and smaller markers
ax1.plot(episode_numbers, total_rewards, marker='o', linestyle='-', color='b', markersize=3, linewidth=2)
ax1.axhline(y=0, color='g', linestyle='--', label='y=0')
ax1.set_title('Total Reward per Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')

# Plot average rewards with smoother lines and smaller markers
ax2.plot(episode_numbers, avg_rewards, marker='o', linestyle='-', color='r', markersize=3, linewidth=2)
ax2.set_title('Average Reward per Episode')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Reward')

# Adjust spacing between subplots
plt.tight_layout()
plt.savefig('reward_plot_DDQN.jpg', format='jpg')

# Show the plots
plt.show()
