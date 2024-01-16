from collections import deque

import cv2
import numpy as np
import random
import tensorflow as tf
from keras.models import load_model
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


# Function to run testing phase and calculate average score
def test_environment(env, num_episodes, agent, max_frame_count):
    total_rewards = []

    for episode in range(num_episodes):
        env.reset_game()
        state = get_sensor_readings(env.getGameState())
        total_reward = 0
        frame_count = 0

        if max_frame_count is not None:
            while not env.game_over() and frame_count < max_frame_count:
                frame_count += 1
                action = agent.select_action(state.reshape(1, -1))
                reward = env.act(env.getActionSet()[action])

                next_state = get_sensor_readings(env.getGameState())
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)
        else:
            while not env.game_over():
                frame_count += 1
                action = agent.select_action(state.reshape(1, -1))
                reward = env.act(env.getActionSet()[action])

                next_state = get_sensor_readings(env.getGameState())
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

    average_score = np.mean(total_rewards)
    return average_score


# Load the trained model
model = load_model('waterworld_dqn_model.h5')

# Set up the testing environments
env_60_molecules = PLE(WaterWorld(width=768, height=768, num_creeps=60), fps=30, display_screen=True)
env_60_molecules.init()

# Create DQNAgent for testing
action_space = len(env_60_molecules.getActionSet())
agent = DQNAgent(model, model, action_space)  # Both models are the same for testing

# Test the model with the DQNAgent in an environment with 60 molecules
test_episodes_60_molecules = 10
average_score_60_molecules = test_environment(env_60_molecules, test_episodes_60_molecules, agent, 1000)
print(f"Average Score in Environment with 60 Molecules: {average_score_60_molecules}")

# Set up the testing environments
env_5_molecules = PLE(WaterWorld(width=768, height=768, num_creeps=5), fps=30, display_screen=True)
env_5_molecules.init()

# Test the model with the DQNAgent in an environment with 60 molecules
test_episodes_5_molecules = 10
average_score_5_molecules = test_environment(env_5_molecules, test_episodes_5_molecules, agent, None)
print(f"Average Score in Environment with 5 Molecules: {average_score_5_molecules}")
