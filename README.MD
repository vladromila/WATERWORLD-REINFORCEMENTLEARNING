# Benchmark Performance: Waterworld DDQN Agent

## Introduction
This repository contains the implementation and benchmarking results of a Double Deep Q-Network (DDQN) agent designed for the WaterWorld game. The DDQN agent was evaluated on its ability to navigate strategically, balancing the collection of beneficial green molecules against harmful red ones.

## Techniques Used
The implemented DDQN agent incorporates a main neural network for action selection and a target network for calculating stable Q-values. The sensory inputs comprise the relative positions and types of nearby molecules (green or red) in relation to the player molecule. The agent's learning process is enriched by a replay buffer that stores a history of experiences. During training, the agent's policy is improved iteratively by learning from both recent and past experiences.

## Evaluation Metrics

- **Total Reward per Episode**: This metric sums up all rewards (positive for green molecules and negative for red molecules) that the agent collects during an episode.
- **Average Reward per Episode**: This represents the average of total rewards over all episodes, indicating the agent's performance consistency over time.

![plot](https://github.com/vladromila/WATERWORLD-REINFORCEMENTLEARNING/blob/main/plot.jpeg?raw=true)

## Training Analysis
The DDQN agent demonstrated a significant learning capability over 2500 episodes, as evidenced by the average training score of 10.21. The variability in total rewards per episode suggests the exploration-exploitation trade-off inherent in training. The steady increase in average reward per episode indicates a consistent improvement in the agent's policy.

## Testing Analysis

An average score of 9.7 was obtained in 10 episodes with 60 Molecules (the episode finished when number of frames reached 1000). A notable result was represented by an average score of 12.5 which was obtained in 10 episodes with 5 molecules (the
episode finished when the game finished).

## Benchmark Performance Insights
- The agent showed robustness in testing, achieving scores that closely matched or exceeded the training phase's latter average reward.
- A higher average score was achieved in environments with fewer molecules, suggesting more efficient decision-making in less crowded scenarios.
- Some episodes exhibited room for optimization, particularly in unfamiliar environmental setups, indicating potential areas for further model refinement.