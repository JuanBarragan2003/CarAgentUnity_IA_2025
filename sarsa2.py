import os
import pickle
import numpy as np
import itertools
import pandas as pd
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class AgenteSARSAUnityFourierMejorado:
    def __init__(self, entorno, alfa=0.0005, gamma=0.95, epsilon=0.9, epsilon_min=0.05, 
                 decaimiento=0.9995, orden_fourier=7, max_bases=5, buffer_normalizacion=1000):
        self.entorno = entorno
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_inicial = epsilon
        self.epsilon_min = epsilon_min
        self.decaimiento = decaimiento
        self.orden_fourier = orden_fourier
        self.buffer_normalizacion = buffer_normalizacion

        # Obtener información del entorno
        self.behavior_name = list(entorno.behavior_specs.keys())[0]
        self.behavior_spec = entorno.behavior_specs[self.behavior_name]

        # Configurar acciones y observaciones
        self.accion_branches = self.behavior_spec.action_spec.discrete_branches
        self.obs_specs = self.behavior_spec.observation_specs
        self.dim_estado = sum([np.prod(obs.shape) for obs in self.obs_specs])
        
        # Generar todas las acciones posibles
        self.todas_las_acciones = self._obtener_todas_las_acciones_posibles()
        print(f"Acciones posibles: {len(self.todas_las_acciones)}")
        print(f"Dimensión del estado: {self.dim_estado}")

        # Buffer para normalización adaptativa
        self.buffer_estados = deque(maxlen=buffer_normalizacion)
        
        # Generar bases de Fourier más eficientemente
        self._generar_bases_fourier_mejoradas(max_bases)
        
        # Inicializar pesos con inicialización Xavier
        self.theta = {}
        for accion in self.todas_las_acciones:
            accion_tuple = tuple(accion)
            # Inicialización Xavier para mejor convergencia
            limite = np.sqrt(6.0 / (len(self.bases) + 1))
            self.theta[accion_tuple] = np.random.uniform(-limite, limite, len(self.bases))
        
        # Scaler para normalización
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        
        # Métricas para monitoreo
        self.q_values_history = deque(maxlen=1000)
        self.td_errors_history = deque(maxlen=1000)
        
        print(f"Agente inicializado con {len(self.bases)} bases de Fourier")

    def _generar_bases_fourier_mejoradas(self, max_bases):
        """Genera bases de Fourier de manera más eficiente y balanceada"""
        bases_list = []
        
        # Agregar la base cero primero (importante para la aproximación)
        base_cero = np.zeros(self.dim_estado, dtype=int)
        bases_list.append(base_cero)
        
        # Generar bases por orden de complejidad
        for orden in range(1, self.orden_fourier + 1):
            # Bases de un solo componente
            for i in range(min(self.dim_estado, 20)):  # Limitar dimensiones para evitar explosión
                base = np.zeros(self.dim_estado, dtype=int)
                base[i] = orden
                if len(bases_list) < max_bases:
                    bases_list.append(base.copy())
            
            # Bases de múltiples componentes (solo orden 1 para evitar explosión)
            if orden == 1 and self.dim_estado <= 20:
                for i in range(min(self.dim_estado, 10)):
                    for j in range(i+1, min(self.dim_estado, 15)):
                        base = np.zeros(self.dim_estado, dtype=int)
                        base[i] = 1
                        base[j] = 1
                        if len(bases_list) < max_bases:
                            bases_list.append(base.copy())
        
        self.bases = np.array(bases_list[:max_bases])
        print(f"Generadas {len(self.bases)} bases de Fourier (orden máximo: {self.orden_fourier})")

    def _estado_a_vector(self, observaciones):
        """Convierte observaciones en vector de estado normalizado con manejo robusto"""
        # Concatenar todas las observaciones
        if len(observaciones) == 2:
            obs1 = observaciones[0].flatten()
            obs2 = observaciones[1].flatten()
            estado = np.concatenate([obs1, obs2])
        else:
            estado = observaciones[0].flatten()
        
        # Asegurar dimensión correcta
        if len(estado) != self.dim_estado:
            if len(estado) > self.dim_estado:
                estado = estado[:self.dim_estado]
            else:
                estado = np.pad(estado, (0, self.dim_estado - len(estado)))
        
        # Clip valores extremos y manejar NaN/inf
        estado = np.nan_to_num(estado, nan=0.0, posinf=10.0, neginf=-10.0)
        estado = np.clip(estado, -90, 90)
        
        # Agregar al buffer para normalización adaptativa
        self.buffer_estados.append(estado.copy())
        
        # Normalización adaptativa
        if len(self.buffer_estados) >= 10000:  # Mínimo de muestras para normalizar
            if not self._scaler_fitted or len(self.buffer_estados) % 100 == 0:
                # Re-entrenar scaler periódicamente
                buffer_array = np.array(list(self.buffer_estados))
                self.scaler.fit(buffer_array)
                self._scaler_fitted = True
        
        if self._scaler_fitted:
            estado_normalizado = self.scaler.transform(estado.reshape(1, -1))[0]
            # Clip después de normalización para evitar valores extremos
            estado_normalizado = np.clip(estado_normalizado, -3, 3)
        else:
            # Normalización simple si no hay suficientes datos
            estado_normalizado = estado / (np.std(estado) + 1e-8)
            estado_normalizado = np.clip(estado_normalizado, -3, 3)
        
        return estado_normalizado

    def _phi(self, estado_normalizado):
        """Características de Fourier mejoradas"""
        # Calcular productos punto
        productos = np.dot(self.bases, estado_normalizado)
        
        # Usar tanto coseno como seno para mayor capacidad de aproximación
        features_cos = np.cos(np.pi * productos)
        
        # Para orden > 1, agregar algunas características seno
        if self.orden_fourier > 1:
            # Solo para algunas bases para no duplicar dimensión
            n_sin_features = min(len(self.bases) // 3, 90)
            productos_sin = productos[:n_sin_features]
            features_sin = np.sin(np.pi * productos_sin)
            features = np.concatenate([features_cos, features_sin])
        else:
            features = features_cos
        
        return features

    def _obtener_q(self, estado, accion):
        """Calcula Q(s,a) con regularización"""
        accion_tuple = tuple(accion)
        phi_s = self._phi(estado)
        
        # Asegurar compatibilidad de dimensiones
        if len(self.theta[accion_tuple]) != len(phi_s):
            # Ajustar theta si es necesario
            if len(self.theta[accion_tuple]) < len(phi_s):
                # Extender theta con ceros
                extension = np.zeros(len(phi_s) - len(self.theta[accion_tuple]))
                self.theta[accion_tuple] = np.concatenate([self.theta[accion_tuple], extension])
            else:
                # Truncar phi_s
                phi_s = phi_s[:len(self.theta[accion_tuple])]
        
        q_val = np.dot(self.theta[accion_tuple], phi_s)
        
        # Agregar a historial para monitoreo
        self.q_values_history.append(abs(q_val))
        
        return q_val

    def obtener_accion(self, observaciones, entrenar=True):
        """Selecciona acción usando ε-greedy mejorado"""
        estado = self._estado_a_vector(observaciones)

        # Exploración con decaimiento adaptativo
        if entrenar and np.random.rand() < self.epsilon:
            accion_idx = np.random.randint(len(self.todas_las_acciones))
            return list(self.todas_las_acciones[accion_idx])

        # Explotación con manejo robusto
        q_values = []
        for accion in self.todas_las_acciones:
            try:
                q_val = self._obtener_q(estado, accion)
                q_values.append(q_val)
            except:
                q_values.append(0.0)  # Valor por defecto si hay error
        
        # Seleccionar mejor acción
        mejor_idx = np.argmax(q_values)
        return list(self.todas_las_acciones[mejor_idx])

    def actualizar(self, observaciones, accion, recompensa, siguientes_observaciones, terminado):
        """Actualización SARSA mejorada con regularización y clipping"""
        s = self._estado_a_vector(observaciones)
        phi_s = self._phi(s)
        accion_tuple = tuple(accion)
        
        # Asegurar compatibilidad dimensional
        if len(self.theta[accion_tuple]) != len(phi_s):
            if len(self.theta[accion_tuple]) < len(phi_s):
                extension = np.zeros(len(phi_s) - len(self.theta[accion_tuple]))
                self.theta[accion_tuple] = np.concatenate([self.theta[accion_tuple], extension])
            else:
                phi_s = phi_s[:len(self.theta[accion_tuple])]
        
        # Q(s,a) actual
        q_sa = np.dot(self.theta[accion_tuple], phi_s)

        if not terminado:
            # Obtener siguiente acción
            siguiente_accion = self.obtener_accion(siguientes_observaciones, entrenar=True)
            s_next = self._estado_a_vector(siguientes_observaciones)
            q_next = self._obtener_q(s_next, siguiente_accion)
        else:
            q_next = 0.0

        # Clipping de recompensa para estabilidad
        recompensa = np.clip(recompensa, -1, 1)
        
        # Actualización SARSA con regularización
        objetivo = recompensa + self.gamma * q_next
        error_td = objetivo - q_sa
        
        # Agregar error TD al historial
        self.td_errors_history.append(abs(error_td))
        
        # Clipping del error TD para evitar updates extremos
        error_td = np.clip(error_td, -5, 5)
        
        # Learning rate adaptativo basado en magnitud del error
        alfa_adaptativo = self.alfa
        if abs(error_td) > 2.0:
            alfa_adaptativo = self.alfa * 0.5  # Reducir learning rate para errores grandes
        
        # Actualizar pesos con regularización L2
        regularizacion = 1e-6
        gradiente = alfa_adaptativo * error_td * phi_s
        self.theta[accion_tuple] += gradiente - regularizacion * self.theta[accion_tuple]
        
        # Clipping de pesos para evitar valores extremos
        self.theta[accion_tuple] = np.clip(self.theta[accion_tuple], -100, 100)

    def _obtener_todas_las_acciones_posibles(self):
        """Genera todas las combinaciones de acciones posibles"""
        acciones = []
        for combinacion in itertools.product(*[range(branch) for branch in self.accion_branches]):
            acciones.append(combinacion)
        return acciones

    def decaer_epsilon(self):
        """Decaimiento de epsilon mejorado"""
        self.epsilon = max(self.epsilon * self.decaimiento, self.epsilon_min)

    def reiniciar_epsilon(self, factor=0.5):
        """Reinicia epsilon para exploración adicional si es necesario"""
        self.epsilon = max(self.epsilon_inicial * factor, self.epsilon_min)

    def obtener_estadisticas(self):
        """Obtiene estadísticas del entrenamiento"""
        stats = {
            'epsilon': self.epsilon,
            'q_values_mean': np.mean(self.q_values_history) if self.q_values_history else 0,
            'q_values_std': np.std(self.q_values_history) if self.q_values_history else 0,
            'td_errors_mean': np.mean(self.td_errors_history) if self.td_errors_history else 0,
            'td_errors_std': np.std(self.td_errors_history) if self.td_errors_history else 0,
            'pesos_norm': np.mean([np.linalg.norm(pesos) for pesos in self.theta.values()])
        }
        return stats

    def guardar(self, archivo="modelo_fourier_mejorado.pkl"):
        """Guarda el modelo"""
        modelo_data = {
            'theta': self.theta,
            'bases': self.bases,
            'scaler': self.scaler,
            'scaler_fitted': self._scaler_fitted,
            'dim_estado': self.dim_estado,
            'todas_las_acciones': self.todas_las_acciones,
            'buffer_estados': list(self.buffer_estados),
            'hiperparametros': {
                'alfa': self.alfa,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_inicial': self.epsilon_inicial,
                'orden_fourier': self.orden_fourier
            }
        }
        
        with open(archivo, "wb") as f:
            pickle.dump(modelo_data, f)
        print(f"Modelo guardado en {archivo}")

    def cargar(self, archivo="modelo_fourier_mejorado.pkl"):
        """Carga el modelo"""
        if os.path.exists(archivo):
            with open(archivo, "rb") as f:
                modelo_data = pickle.load(f)
            
            self.theta = modelo_data['theta']
            self.bases = modelo_data['bases']
            self.scaler = modelo_data['scaler']
            self._scaler_fitted = modelo_data['scaler_fitted']
            self.dim_estado = modelo_data['dim_estado']
            self.todas_las_acciones = modelo_data['todas_las_acciones']
            
            if 'buffer_estados' in modelo_data:
                self.buffer_estados = deque(modelo_data['buffer_estados'], 
                                          maxlen=self.buffer_normalizacion)
            
            if 'hiperparametros' in modelo_data:
                hiper = modelo_data['hiperparametros']
                self.alfa = hiper['alfa']
                self.gamma = hiper['gamma']
                self.epsilon = hiper['epsilon']
                if 'epsilon_inicial' in hiper:
                    self.epsilon_inicial = hiper['epsilon_inicial']
                self.orden_fourier = hiper['orden_fourier']
            
            print(f"Modelo cargado desde {archivo}")
        else:
            print(f"No se encontró el archivo {archivo}")


def entrenar_agente_mejorado(puerto=5004, episodios=3000):
    """Función de entrenamiento mejorada con diagnósticos"""
    try:
        entorno = UnityEnvironment(file_name=None, base_port=puerto, seed=42, side_channels=[])
        entorno.reset()
        
        # Parámetros optimizados
        agente = AgenteSARSAUnityFourierMejorado(
            entorno, 
            alfa=0.005,  # Learning rate más conservador
            gamma=0.99999,  # Factor de descuento alto
            epsilon=0.9999,  # Exploración inicial moderada
            epsilon_min=0.001,  # Mínimo de exploración
            decaimiento=0.999,  # Decaimiento más lento
            orden_fourier=3,  # Orden conservador
            max_bases=100,  # Menos bases para evitar overfitting
            buffer_normalizacion=500
        )

        historico = []
        mejor_recompensa = float('-inf')
        recompensas_recientes = deque(maxlen=100)
        sin_mejora_contador = 0

        for episodio in range(episodios):
            entorno.reset()
            decision_steps, terminal_steps = entorno.get_steps(agente.behavior_name)
            
            if len(decision_steps) == 0:
                print(f"No hay agentes activos en episodio {episodio + 1}")
                continue
                
            recompensa_total = 0
            pasos = 0
            max_pasos = 3000  # Límite más conservador

            while len(decision_steps) > 0 and pasos < max_pasos:
                agent_id = decision_steps.agent_id[0]
                observaciones = [obs[0] for obs in decision_steps.obs]

                # Seleccionar acción
                acciones = agente.obtener_accion(observaciones)
                action_tuple = ActionTuple(discrete=np.array([acciones]))
                entorno.set_actions(agente.behavior_name, action_tuple)
                entorno.step()

                # Obtener siguiente estado
                prev_decision_steps = decision_steps
                decision_steps, terminal_steps = entorno.get_steps(agente.behavior_name)

                # Procesar resultado
                if len(terminal_steps) > 0 and agent_id in terminal_steps.agent_id:
                    idx = np.where(terminal_steps.agent_id == agent_id)[0][0]
                    siguientes_observaciones = [obs[idx] for obs in terminal_steps.obs]
                    recompensa = terminal_steps.reward[idx]
                    terminado = True
                elif len(decision_steps) > 0 and agent_id in decision_steps.agent_id:
                    idx = np.where(decision_steps.agent_id == agent_id)[0][0]
                    siguientes_observaciones = [obs[idx] for obs in decision_steps.obs]
                    recompensa = decision_steps.reward[idx]
                    terminado = False
                else:
                    terminado = True
                    recompensa = 0
                    siguientes_observaciones = observaciones

                # Actualizar agente
                agente.actualizar(observaciones, acciones, recompensa, 
                                siguientes_observaciones, terminado)
                
                recompensa_total += recompensa
                pasos += 1

                if terminado:
                    break

            # Decaer epsilon
            agente.decaer_epsilon()
            recompensas_recientes.append(recompensa_total)

            # Diagnósticos y adaptaciones
            if (episodio + 1) % 50 == 0:
                stats = agente.obtener_estadisticas()
                promedio_reciente = np.mean(recompensas_recientes) if recompensas_recientes else 0
                
                print(f"Episodio {episodio + 1}/{episodios}")
                print(f"  Recompensa: {recompensa_total:.2f} (Promedio últimos 100: {promedio_reciente:.2f})")
                print(f"  Pasos: {pasos}, Epsilon: {agente.epsilon:.4f}")
                print(f"  Q-values medio: {stats['q_values_mean']:.3f}")
                print(f"  TD error medio: {stats['td_errors_mean']:.3f}")
                print(f"  Norma pesos: {stats['pesos_norm']:.3f}")
                
                # Reiniciar exploración si no hay mejora
                if promedio_reciente <= mejor_recompensa * 0.8 and episodio > 500:
                    sin_mejora_contador += 1
                    if sin_mejora_contador >= 3:  # Sin mejora por 150 episodios
                        print("  -> Reiniciando exploración")
                        agente.reiniciar_epsilon(factor=0.3)
                        sin_mejora_contador = 0
                else:
                    sin_mejora_contador = 0

            # Guardar mejor modelo
            promedio_actual = np.mean(list(recompensas_recientes)[-50:]) if len(recompensas_recientes) >= 50 else recompensa_total
            if promedio_actual > mejor_recompensa:
                mejor_recompensa = promedio_actual
                agente.guardar("mejor_modelo_sarsa_mejorado.pkl")

            # Guardar modelo periódicamente
            if (episodio + 1) % 1000 == 0:
                agente.guardar(f"modelo_sarsa_mejorado_ep{episodio+1}.pkl")

            # Guardar métricas
            stats = agente.obtener_estadisticas()
            historico.append({
                "episodio": episodio + 1,
                "recompensa_total": recompensa_total,
                "pasos": pasos,
                "epsilon": agente.epsilon,
                "q_values_mean": stats['q_values_mean'],
                "td_errors_mean": stats['td_errors_mean']
            })

        # Guardado final
        agente.guardar("modelo_sarsa_mejorado_final.pkl")
        
        # Guardar métricas
        df = pd.DataFrame(historico)
        df.to_csv("metricas_sarsa_mejorado.csv", index=False)
        print("Métricas guardadas en metricas_sarsa_mejorado.csv")
        
        entorno.close()

    except Exception as e:
        print(f"Error durante entrenamiento: {e}")
        if 'entorno' in locals():
            entorno.close()


if __name__ == "__main__":
    puerto = 5004
    print("Iniciando entrenamiento mejorado...")
    entrenar_agente_mejorado(puerto, episodios=3000)