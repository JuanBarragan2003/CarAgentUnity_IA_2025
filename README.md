# CarAgentUnity_IA_2025

# 🚗 CarAgentUnity_IA_2025

Este repositorio contiene un agente inteligente de conducción autónoma desarrollado en **Unity** con soporte para **ML-Agents** y algoritmos personalizados como **PPO** de la herramienta de la MLagents, **SARSA** y **DQL**. El agente aprende a recorrer un circuito cerrado pasando por checkpoints ordenados, evitando obstáculos y maximizando la recompensa.
---
## 🎯 Objetivos del Proyecto

- Desarrollar un entorno de simulación de conducción en Unity.
- Entrenar un agente usando algoritmos de aprendizaje por refuerzo.
- Implementar y comparar modelos con:
  - **ML-Agents (PPO)**
  - **SARSA** (con aprocimaciones mediante Fourier)
  - **DQL clásico**
- Evaluar el comportamiento con métricas detalladas y gráficas.

---

## 📁 Estructura del Repositorio

```plaintext
CarAgentUnity_IA_2025/
│
├── Scripts/
│   ├── CarAgent.cs           # Script del agente para PPO 
│   ├── EnvironmentManager.cs # Reinicio y lógica del entorno
│   ├── HUDManager.cs         # HUD con métricas visuales
│   └── RaySensorGizmo.cs     # Visualización de raycasts
│   ├── Prefacs               # Escena principal del entorno
│   ├── train_sarsa.py         # Entrenamiento SARSA personalizado
│   ├── train_dql.py            # Entrenamiento DQL clásico
│   ├── sarsa.py        # Codificación de Fourier para observaciones
│  
│
├── Results_SARSA/
│   ├── episodic_rewards.csv
│   ├── reward_vs_steps.png
│   └── checkpoints_plot.png
│
├── Results_PPO/
│   ├── summaries/
│   ├── plots/
│   └── training_logs/
│   ├── config/
│       └── car_config.yaml   # Configuración de entrenamiento PPO

```

***

## 🧪 Metodología de Entrenamiento

El entrenamiento del agente fue realizado usando dos enfoques complementarios:

### 🔁 1. Entrenamiento con PPO (Proximal Policy Optimization)

Utilizamos **ML-Agents** de Unity para entrenar al agente directamente dentro del entorno 3D.

**Pasos realizados:**

1. Definimos el comportamiento del agente en `Agent.cs`, utilizando observaciones como raycasts, velocidad, y distancia al checkpoint.
2. Instalamos la herramienta MLagents de unity. 
3. Configuramos los parámetros de entrenamiento en el archivo `car_config.yaml`, incluyendo hiperparámetros como:
   - `learning_rate = 0.0003`
   - `buffer_size = 10240`
   - `batch_size = 1024`
   - `gamma = 0.99`, entre otros.
4. Ejecutamos el entorno desde Unity y conectamos ML-Agents con:
   ```bash
   mlagents-learn config/car_config.yaml --run-id=ppo_run_1
 ```
 ```
***

## 🔂 Entrenamiento con SARSA

Implementamos un entrenamiento personalizado utilizando el algoritmo **SARSA (State-Action-Reward-State-Action)**, ideal para entornos con estados discretizados o codificados, como es el caso de este proyecto, donde usamos una base de funciones de **Fourier** para representar las observaciones continuas del entorno.

### 📋 Descripción del algoritmo

SARSA actualiza su tabla Q en función de la política actual. A diferencia de Q-Learning, aprende en tiempo real de la secuencia completa (estado, acción, recompensa, siguiente estado, siguiente acción), lo que lo hace más estable en entornos no deterministas como Unity.

### 🧩 Observaciones utilizadas

- Distancias a obstáculos desde sensores Raycast.
- Velocidad del vehículo.
- Distancia al siguiente checkpoint.
- Número de checkpoint alcanzado.

Estas observaciones se codifican mediante Fourier para convertirlas en una representación más compacta y generalizable.

### ⚙️ Entrenamiento paso a paso

1. Asegúrate de tener Unity corriendo en modo "External Communicator".
2. Ejecuta el entrenamiento con:
   ```bash
   python  scripts/sarsa.py --episodes 5000 --steps 3000 --gamma 0.99 --alpha 0.001 --epsilon 0.999




