from collections import deque

import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from ple import PLE
from ple.games.waterworld import WaterWorld


# Define the Deep Q-Network (DQN) model
def build_dqn_model(num_sensors, max_molecules, num_actions):
    input_size = num_sensors * max_molecules * 3
    sensor_input = keras.Input(shape=(input_size,))

    # Dense layers for processing sensor data and positions
    dense1 = keras.layers.Dense(256, activation='relu')(sensor_input)
    dense2 = keras.layers.Dense(256, activation='relu')(dense1)
    output = keras.layers.Dense(num_actions)(dense2)

    model = keras.Model(inputs=sensor_input, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

    return model


# Image preprocessing
def preprocess_screen(screen):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


# Correct the sensor readings function
def get_sensor_readings(state, num_sensors=8, max_molecules=10, detection_radius=200):
    sensor_readings = np.zeros(num_sensors * max_molecules)  # Initialize readings
    molecule_positions = np.zeros((num_sensors, max_molecules, 2))  # Initialize positions

    player_x, player_y = state['player_x'], state['player_y']
    molecule_idx = 0  # Index to keep track of the number of molecules

    for creep_type in ['GOOD', 'BAD']:
        for pos in state['creep_pos'][creep_type]:
            dx, dy = pos[0] - player_x, pos[1] - player_y
            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance <= detection_radius:
                sensor_value = 1 if creep_type == 'GOOD' else -1
                sensor_readings[molecule_idx] = sensor_value

                # Normalize and store molecule positions
                normalized_dx = dx / detection_radius
                normalized_dy = dy / detection_radius
                molecule_positions[molecule_idx // max_molecules, molecule_idx % max_molecules] = (normalized_dx, normalized_dy)
                molecule_idx += 1

    # Flatten the positions array and concatenate with sensor readings
    positions_flat = molecule_positions.flatten()
    full_sensor_input = np.concatenate((sensor_readings, positions_flat))

    # Ensure the full sensor input is the correct length
    expected_length = num_sensors * max_molecules * 3
    if len(full_sensor_input) < expected_length:
        # Pad with zeros if the input is shorter than expected
        full_sensor_input = np.pad(full_sensor_input, (0, expected_length - len(full_sensor_input)), 'constant')

    return full_sensor_input

# Define the Deep Q-Network Agent
class DQNAgent:
    def __init__(self, model, target_model, action_space, replay_buffer_size=10000, epsilon_initial=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99):
        self.model = model
        self.target_model = target_model
        self.action_space = action_space
        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Reshape state and next_state to add the batch dimension
            state = np.array([state])
            next_state = np.array([next_state])

            target_f = self.model.predict(state)
            if done:
                target_f[0][action] = reward
            else:
                # Use the target model to predict the Q-value for the next state and take max
                Q_future = max(self.target_model.predict(next_state)[0])
                target_f[0][action] = reward + Q_future * self.gamma

            # Train the model with the state as input and target_f as the target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# Set up the environment and action space
game = WaterWorld(width=512, height=512, num_creeps=60)
env = PLE(game, fps=30, display_screen=False)
env.init()
action_space = len(env.getActionSet())

# Initialize models and agent
num_sensors = 8  # Number of sensors
max_molecules = 10  # Maximum number of molecules to consider
model = build_dqn_model(num_sensors, max_molecules, action_space)
target_model = build_dqn_model(num_sensors, max_molecules, action_space)

agent = DQNAgent(model, target_model, action_space, replay_buffer_size=10000)

update_target_frequency = 100  # Define frequency of updating target model
batch_size = 32  # Batch size for replay

# Training loop
num_episodes = 2500
replay_interval = 50
max_frame_count = 500
total_global_reward = 0

for episode in range(num_episodes):
    env.reset_game()
    state = get_sensor_readings(game.getGameState(), num_sensors, max_molecules)

    total_reward = 0
    frame_count = 0
    while not env.game_over() and frame_count < max_frame_count:
        frame_count += 1
        action = agent.select_action(state.reshape(1, -1))  # Reshape state for network input
        reward = env.act(env.getActionSet()[action])

        next_state = get_sensor_readings(game.getGameState(), num_sensors, max_molecules)
        done = env.game_over()

        agent.remember(state, action, reward, next_state, done)

        if frame_count % replay_interval == 0:
            agent.replay(batch_size)

        state = next_state
        total_reward += reward

        if frame_count % update_target_frequency == 0:
            agent.update_target_model()

    agent.update_epsilon()
    total_global_reward += total_reward

    # Print metrics
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Avg Reward per episode: {total_global_reward / (episode + 1)}")

# Save the trained model
model.save('waterworld_dqn_model.h5')
