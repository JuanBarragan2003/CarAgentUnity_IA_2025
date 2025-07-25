import numpy as np
import random
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry
from mlagents_envs.environment import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import pickle

# Parámetros de Fourier
FOURIER_ORDER = 3  # Ajusta este valor si necesitas más resolución
ALPHA = 0.01       # Tasa de aprendizaje
GAMMA = 0.99       # Factor de descuento
EPSILON = 0.1      # Exploración
NUM_EPISODES = 200

class FourierQLearningAgent:
    def __init__(self, env, behavior_name, order=FOURIER_ORDER):
        self.behavior_name = behavior_name
        self.order = order
        spec = env.behavior_specs[behavior_name]
        self.num_actions = spec.action_spec.discrete_branches[0]
        self.obs_dim = spec.observation_specs[0].shape[0]

        self.coefficients = self.create_fourier_basis(self.obs_dim, self.order)
        self.num_basis = len(self.coefficients)
        self.weights = np.zeros((self.num_actions, self.num_basis))

    def create_fourier_basis(self, dim, order):
        from itertools import product
        return np.array([np.array(t) for t in product(range(order + 1), repeat=dim)])

    def get_features(self, state):
        return np.cos(np.pi * np.dot(self.coefficients, state))

    def get_q_values(self, state):
        features = self.get_features(state)
        return np.dot(self.weights, features)

    def select_action(self, state):
        if random.random() < EPSILON:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        td_target = reward + GAMMA * np.max(next_q_values)
        td_error = td_target - q_values[action]
        features = self.get_features(state)
        self.weights[action] += ALPHA * td_error * features

    def save(self, path="qlearning_model.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

def entrenar_agente():
    # Configuración para correr en modo editor
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=1.0)
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        raise Exception("No se detectaron behavior_specs. Asegúrate de que el entorno esté correctamente inicializado.")
    behavior_name = behavior_names[0]

    agente = FourierQLearningAgent(env, behavior_name)

    for episodio in range(NUM_EPISODES):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(decision_steps) == 0:
            continue

        agent_id = list(decision_steps.agent_id)[0]
        obs = decision_steps[agent_id].obs[0]
        state = np.array(obs, dtype=np.float32)

        done = False
        episodic_reward = 0

        while not done:
            action = agente.select_action(state)
            action_tuple = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if agent_id in terminal_steps:
                next_obs = terminal_steps[agent_id].obs[0]
                reward = terminal_steps[agent_id].reward
                done = True
            else:
                next_obs = decision_steps[agent_id].obs[0]
                reward = decision_steps[agent_id].reward
                done = False

            next_state = np.array(next_obs, dtype=np.float32)
            agente.update(state, action, reward, next_state)
            state = next_state
            episodic_reward += reward

        print(f"Episodio {episodio+1}/{NUM_EPISODES} - Recompensa: {episodic_reward:.2f}")

    agente.save()
    print("Entrenamiento completado y modelo guardado.")
    env.close()

if __name__ == "__main__":
    entrenar_agente()
