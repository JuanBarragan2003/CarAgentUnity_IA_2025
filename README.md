# CarAgentUnity_IA_2025

# 🚗 CarAgentUnity_IA_2025

Este repositorio contiene un agente inteligente de conducción autónoma desarrollado en **Unity** con soporte para **ML-Agents** y algoritmos personalizados como **PPO** de la herramienta de la MLagents, **SARSA** y **DQL**. El agente aprende a recorrer un circuito cerrado pasando por checkpoints ordenados, evitando obstáculos y maximizando la recompensa.

---

## 🎯 Objetivos del Proyecto

- Desarrollar un entorno de simulación de conducción en Unity.
- Entrenar un agente usando algoritmos de aprendizaje por refuerzo.
- Implementar y comparar modelos con:
  - **ML-Agents (PPO)**
  - **SARSA** (con y sin Fourier)
  - **DQL clásico**
- Evaluar el comportamiento con métricas detalladas y gráficas.

---

## 📁 Estructura del Repositorio

```plaintext
CarAgentUnity_IA_2025/
│
├── Assets/
│   ├── Scripts/
│   │   ├── CarAgent.cs           # Comportamiento del agente
│   │   ├── EnvironmentManager.cs # Reinicio y lógica del entorno
│   │   ├── HUDManager.cs         # HUD con métricas visuales
│   │   └── RaySensorGizmo.cs     # Visualización de raycasts
│   ├── Scenes/
│   │   └── Main.unity            # Escena principal del entorno
│
├── Python/
│   ├── train_sarsa.py           # Entrenamiento SARSA personalizado
│   ├── train_dql.py             # Entrenamiento DQL clásico
│   ├── fourier_basis.py         # Codificación de Fourier para observaciones
│   └── analyze_results.py       # Gráficas y análisis de resultados
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
│
├── config/
│   └── car_config.yaml          # Configuración de entrenamiento PPO
│
└── README.md
