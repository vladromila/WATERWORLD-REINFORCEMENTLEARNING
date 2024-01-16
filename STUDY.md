# DDQN Agent Ablation Study in Waterworld

## Study Overview
In this ablation study, we investigate the efficacy of a Double Deep Q-Network (DDQN) agent within the Waterworld environment. Our study focuses on key components such as neural network architecture, replay buffer usage, and target network update frequency, to understand their individual and collective impacts on the agent's performance.

## Key Components
- **Neural Network Architecture**: We utilize a dual-network setup with each network comprising two dense layers with 256 neurons each, using ReLU activation functions.
- **Replay Buffer**: A replay buffer with a capacity of 10,000 experiences is employed, allowing the agent to learn from a diversified set of past experiences, enhancing learning efficiency.
- **Target Network Update**: The target network, which helps in stabilizing the training process, is updated every 100 steps to reflect the main network's weights.

## Agent Behavior and Training Process
The DDQN agent was trained over 2,500 episodes with a frame cap of 1,000 per episode. A significant improvement in decision-making and stability was observed compared to traditional DQN models. Key metrics such as average score per episode and total score per episode were used to monitor performance.

## Findings
- **Performance**: The agent achieved an average training score of 10.21 and testing scores of 9.7 and 12.5 in environments with 60 and 5 molecules, respectively.
- **Influential Components**: The replay buffer size and the frequency of target network updates were found to be the most impactful components on the agent's learning progress.
- **Efficiency**: The architecture was found to be efficient enough to be trained on a CPU while still achieving robust results.

## Conclusions and Future Work
The DDQN agent demonstrates enhanced decision-making and stability, with the replay buffer and dual-network architecture being key to its success. Future work could explore prioritized experience replay, different network architectures, and transfer learning to adapt the agent to new environments.

