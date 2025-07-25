import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from collections import deque
import random
import time

# === Hiperparámetros ===
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 20000  
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 500
MAX_STEPS =  3000
SAVE_MODEL_PATH = "DQLCarAgent2.onnx"
RESET_EVERY = 3000
MAX_EPISODE_TIME = 90.0  # en segundos
FIXED_TIMESTEP = 0.02  # Mismo que en la configuración de Unity
DECISION_PERIOD = 2    # Mismo que en el DecisionRequester de tu agente

# === Red Neuronal ===
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# === Exportar modelo .onnx ===
def export_onnx_model(model, obs_size, save_path):
    dummy_input = torch.randn(1, obs_size)
    torch.onnx.export(model, dummy_input, save_path,
                      export_params=True, opset_version=11,
                      input_names=["input"], output_names=["output"])
    print(f"✅ Modelo guardado en: {save_path}")

# === Inicializa entorno de Unity ===
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=None, side_channels=[channel], base_port=5004)
channel.set_configuration_parameters(time_scale=5.0)
env.reset()

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
obs_size = spec.observation_specs[0].shape[0]
n_actions_branch0 = spec.action_spec.discrete_branches[0]
n_actions_branch1 = spec.action_spec.discrete_branches[1]
total_actions = n_actions_branch0 * n_actions_branch1

def decode_action(index):
    return [index // n_actions_branch1, index % n_actions_branch1]

# === Inicializa red y memoria ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Dispositivo: {device}")
policy_net = QNetwork(obs_size, total_actions).to(device)
target_net = QNetwork(obs_size, total_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# === Bucle de entrenamiento ===
epsilon = EPSILON_START
step_count = 0
episode = 0
episode_start_time = time.time()
episode_reward = 0.0
steps_in_episode = 0
real_step_count = 0

while step_count < MAX_STEPS:
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) == 0:
        env.step()
        continue
    real_step_count += 1

    obs = decision_steps.obs[0][0]
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    if np.random.rand() < epsilon:
        action_index = np.random.randint(total_actions)
    else:
        with torch.no_grad():
            q_values = policy_net(obs_tensor)
            action_index = torch.argmax(q_values).item()

    action = decode_action(action_index)
    action_tuple = spec.action_spec.empty_action(len(decision_steps))
    action_tuple.discrete[:, 0] = action[0]
    action_tuple.discrete[:, 1] = action[1]

    env.set_actions(behavior_name, action_tuple)
    env.step()

    next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)
    agent_id = decision_steps.agent_id[0]

    if agent_id in next_terminal_steps:
        next_step = next_terminal_steps[agent_id]
        done = True
    else:
        next_step = next_decision_steps[agent_id]
        done = False

    reward = next_step.reward
    next_obs = next_step.obs[0]

    episode_reward += reward
    steps_in_episode += 1
    memory.push((obs, action_index, reward, next_obs, done))
    step_count += 1

    if len(memory) >= BATCH_SIZE:
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epsilon = max(EPSILON_END, EPSILON_START - step_count / EPSILON_DECAY)

    if step_count % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    #print(f"Paso {real_step_count} | Epsilon: {epsilon:.3f} | Acción: {action_index} | Recompensa paso: {reward:.2f} | Recompensa acumulada: {episode_reward:.2f}")

    tiempo_actual_estimado = real_step_count * FIXED_TIMESTEP * DECISION_PERIOD

    if done or steps_in_episode >= RESET_EVERY or tiempo_actual_estimado >= MAX_EPISODE_TIME:
        
        print(f"🧠 Episodio #{episode} terminado | Recompensa acumulada: {episode_reward:.2f} | Pasos en episodio: {steps_in_episode} | Pasos reales: {real_step_count}")
        episode += 1
        episode_reward = 0.0
        steps_in_episode = 0
        episode_start_time = time.time()
        real_step_count = 0
        env.reset()
        

# === Finalizar ===
env.close()
export_onnx_model(policy_net, obs_size, SAVE_MODEL_PATH)
print("✅ Entrenamiento terminado")
print(f"Total de pasos: {step_count} | Episodios completados: {episode}")
