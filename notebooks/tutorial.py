"""
Tutorial interactivo de Reinforcement Learning para French Solitaire
====================================================================

Este notebook de Marimo te guiará paso a paso en el proceso de entrenar
un agente de RL para resolver el juego French Solitaire usando PyTorch.

Para ejecutar:
    conda activate french-solitaire
    pip install marimo  # Si no está instalado
    marimo edit tutorial_rl_french_solitaire.py
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # 🎯 Tutorial: Entrenamiento de RL para French Solitaire

        ## Introducción

        Este tutorial te enseñará cómo entrenar un agente de **aprendizaje por refuerzo (RL)** 
        para resolver el juego **French Solitaire** (también conocido como Peg Solitaire o Senku).

        ### ¿Qué aprenderás?

        1. **Conceptos básicos de RL**: estados, acciones, recompensas, política
        2. **Implementación del entorno del juego** compatible con Gymnasium
        3. **Algoritmo DQN** (Deep Q-Network) usando PyTorch
        4. **Entrenamiento y evaluación** del agente
        5. **Tracking de experimentos** con MLflow

        ### Stack tecnológico

        - **PyTorch**: redes neuronales y optimización
        - **Gymnasium**: API estándar para entornos de RL
        - **MLflow**: tracking de experimentos
        - **NumPy**: operaciones matriciales
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    """Configura MLflow (si está disponible) para logging local en ./mlruns."""

    try:
        import mlflow

        # Tracking local por defecto
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("french-solitaire")
        mlflow_available = True
        status_md = "MLflow configurado ✅ — Tracking URI: file:./mlruns, Experimento: 'french-solitaire'"
    except Exception as e:
        mlflow = None
        mlflow_available = False
        status_md = (
            "MLflow no disponible ⚠️ — instala y activa el entorno:\n\n"
            "- conda activate french-solitaire\n"
            "- pip install mlflow\n"
            "Luego re-ejecuta esta celda.\n\n"
            f"Detalle: {e}"
        )

    mo.md(f"""
    ### Registro de experimentos (MLflow)

    {status_md}
    """)
    return mlflow, mlflow_available


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📚 Parte 1: Conceptos básicos de RL

    ### ¿Qué es Reinforcement Learning?

    El aprendizaje por refuerzo es un paradigma de ML donde un **agente** aprende a tomar
    **decisiones** en un **entorno** para maximizar una **recompensa acumulada**.

    #### Componentes clave:

    - **Entorno**: el juego French Solitaire
    - **Estado ($s$)**: configuración actual del tablero (32 posiciones)
    - **Acción ($a$)**: mover una ficha (origen → destino)
    - **Recompensa ($r$)**: feedback numérico tras cada acción
    - **Política ($\pi$)**: estrategia del agente para elegir acciones
    - **Valor $Q(s,a)$**: recompensa esperada al tomar acción $a$ en estado $s$

    ### Reglas del French Solitaire

    ```
    Tablero inicial (7×7):
          O O O
          O O O
      O O O O O O O
      O O O . O O O  ← Centro vacío
      O O O O O O O
          O O O
          O O O

    Objetivo: ¡Dejar solo UNA ficha en el centro!
    ```

    **Movimiento válido**: Saltar una ficha adyacente sobre un espacio vacío
    (horizontal o vertical). La ficha saltada se elimina.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import numpy as np


    # Visualización del tablero inicial
    def create_initial_board():
        """Crea el tablero de French Solitaire (7x7) - estado inicial"""
        board = np.zeros((7, 7), dtype=int)

        # Marcar posiciones válidas (cruz europea)
        valid_positions = [
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 2),
            (5, 3),
            (5, 4),
            (6, 2),
            (6, 3),
            (6, 4),
        ]

        # Colocar fichas (1 = ficha, 0 = vacío/no válido)
        for r, c in valid_positions:
            board[r, c] = 1

        # Centro vacío (posición inicial única vacía)
        board[3, 3] = 0

        return board


    initial_board = create_initial_board()

    mo.md(f"""
    ### Representación del tablero

    ```
    Matriz NumPy (7×7):
    {initial_board}
    ```

    - **1**: ficha presente
    - **0**: espacio vacío (centro) o posición inválida (esquinas)
    - **Total de fichas iniciales**: {np.sum(initial_board)} (objetivo: reducir a 1)
    """)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🏗️ Parte 2: Implementación del entorno (Gymnasium)

    Para entrenar un agente de RL, necesitamos un **entorno** que implemente la API de Gymnasium.

    ### Interfaz requerida

    ```python
    class FrenchSolitaireEnv(gym.Env):
        def __init__(self):
            # Definir espacios de observación y acción
            self.observation_space = ...
            self.action_space = ...

        def reset(self, seed=None):
            # Reiniciar el juego al estado inicial
            # Retorna: observación inicial, info
            ...

        def step(self, action):
            # Ejecutar una acción
            # Retorna: observación, recompensa, terminado, truncado, info
            ...

        def render(self):
            # Visualizar el estado actual (opcional)
            ...
    ```

    ### Diseño de espacios

    #### 1. **Espacio de observación** (estado)

    Opciones:
    - **Matriz plana**: vector de 49 elementos (7×7 aplanado)
    - **Matriz 2D**: tensor de forma (7, 7)
    - **Codificación one-hot**: si queremos distinguir múltiples tipos de celdas

    **Recomendación inicial**: usar matriz plana `Box(low=0, high=1, shape=(49,))`

    #### 2. **Espacio de acción**

    Opciones:
    - **Discreto**: enumerar todos los movimientos posibles (N acciones)
    - **Multidiscreto**: (fila_origen, col_origen, dirección) → 7×7×4 = 196 combinaciones
    - **Tupla**: (origen, destino) con validación

    **Recomendación inicial**: `Discrete(n_max_moves)` con mapeo a (origen, destino)

    ### Función de recompensa

    Crucial para el aprendizaje. Propuesta:

    ```python
    if movimiento_invalido:
        reward = -10  # Penalización fuerte
    elif fichas_restantes == 1:
        reward = +100  # Victoria
    elif fichas_reducidas:
        reward = +1  # Progreso positivo
    else:
        reward = -1  # Costo por movimiento
    ```

    **Trade-off**: recompensas densas (cada paso) vs. sparse (solo al final)
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    import gymnasium as gym
    from gymnasium import spaces


    class SimplifiedFrenchSolitaireEnv(gym.Env):
        """
        Versión simplificada del entorno para demostración educativa.
        En producción, esto estaría en envs/french_solitaire_env.py
        """

        def __init__(self):
            super().__init__()

            # Espacio de observación: tablero 7x7 aplanado (49 elementos)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(49,), dtype=np.float32
            )

            # Espacio de acción: discreto con mapeo FIJO
            # Precomputamos todas las acciones geométricamente posibles (origen + dirección)
            # y las mapeamos a índices estables. Rellenamos hasta 100 con "no-ops" inválidos.
            self.directions = [
                (0, 1),  # derecha
                (0, -1),  # izquierda
                (1, 0),  # abajo
                (-1, 0),  # arriba
            ]
            self.all_actions = []  # lista de tuplas (r_from, c_from, dr, dc)
            valid_set = set()
            for r, c in [
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
                (4, 4),
                (4, 5),
                (4, 6),
                (5, 2),
                (5, 3),
                (5, 4),
                (6, 2),
                (6, 3),
                (6, 4),
            ]:
                valid_set.add((r, c))
            for r, c in valid_set:
                for dr, dc in self.directions:
                    jump_r, jump_c = r + dr, c + dc
                    land_r, land_c = r + 2 * dr, c + 2 * dc
                    if (jump_r, jump_c) in valid_set and (
                        land_r,
                        land_c,
                    ) in valid_set:
                        self.all_actions.append((r, c, dr, dc))
            # Relleno hasta 100 con marcadores inválidos
            self.max_actions = 100
            if len(self.all_actions) > self.max_actions:
                # En caso extremo, truncar para mantener compatibilidad con agente
                self.all_actions = self.all_actions[: self.max_actions]
            self.action_space = spaces.Discrete(self.max_actions)

            # Posiciones válidas en el tablero (cruz europea)
            self.valid_positions = [
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
                (4, 4),
                (4, 5),
                (4, 6),
                (5, 2),
                (5, 3),
                (5, 4),
                (6, 2),
                (6, 3),
                (6, 4),
            ]

            # Inicializar tablero para evitar estados None en análisis estático
            self.board = np.zeros((7, 7), dtype=np.float32)
            self.valid_moves = []  # lista de movimientos válidos (pares (from,to))
            self.action_mask = np.zeros(self.max_actions, dtype=np.int8)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)

            # Crear tablero inicial
            self.board = np.zeros((7, 7), dtype=np.float32)
            for r, c in self.valid_positions:
                self.board[r, c] = 1.0
            self.board[3, 3] = 0.0  # Centro vacío

            self._update_valid_moves()

            observation = self.board.flatten()
            info = {
                "pegs_remaining": int(np.sum(self.board)),
                "action_mask": self.action_mask.copy(),
            }

            return observation, info

        def _update_valid_moves(self):
            """Actualiza lista de movimientos válidos y la máscara de acciones."""
            self.valid_moves = []
            self.action_mask.fill(0)
            for idx, (r_from, c_from, dr, dc) in enumerate(self.all_actions):
                # Acciones de relleno (si las hay) quedan inválidas
                if idx >= self.max_actions:
                    break
                jump_r, jump_c = r_from + dr, c_from + dc
                land_r, land_c = r_from + 2 * dr, c_from + 2 * dc
                if (
                    self.board[r_from, c_from] == 1
                    and self.board[jump_r, jump_c] == 1
                    and self.board[land_r, land_c] == 0
                ):
                    self.action_mask[idx] = 1
                    self.valid_moves.append(((r_from, c_from), (land_r, land_c)))

        def step(self, action):
            """
            Ejecuta una acción (índice de movimiento)
            Retorna: observation, reward, terminated, truncated, info
            """
            # Validar que la acción sea válida segun la máscara
            if action >= self.max_actions or self.action_mask[action] == 0:
                # Acción inválida (índice fuera de rango o no válida en este estado)
                reward = -10.0
                observation = self.board.flatten()
                info = {
                    "pegs_remaining": int(np.sum(self.board)),
                    "valid": False,
                    "action_mask": self.action_mask.copy(),
                }
                return observation, reward, False, False, info

            # Ejecutar movimiento válido (derivado de la acción fija)
            r_from, c_from, dr, dc = self.all_actions[action]
            r_to, c_to = r_from + 2 * dr, c_from + 2 * dc
            r_jump, c_jump = r_from + dr, c_from + dc

            # Aplicar movimiento
            self.board[r_from, c_from] = 0.0
            self.board[r_jump, c_jump] = 0.0
            self.board[r_to, c_to] = 1.0

            pegs_remaining = int(np.sum(self.board))

            # Calcular recompensa (victoria SOLO si la última ficha queda en el centro)
            if pegs_remaining == 1:
                if self.board[3, 3] == 1.0:
                    reward = 150.0  # Victoria válida en el centro
                    terminated = True
                else:
                    reward = -100.0  # Final inválido (no está en el centro)
                    terminated = True
            else:
                reward = 1.0  # Progreso (reducimos una ficha)
                terminated = False

            # Actualizar movimientos válidos
            self._update_valid_moves()

            # Verificar si no hay más movimientos (derrota)
            if len(self.valid_moves) == 0 and pegs_remaining > 1:
                reward = -50.0
                terminated = True

            observation = self.board.flatten()
            info = {
                "pegs_remaining": pegs_remaining,
                "valid": True,
                "moves_available": int(self.action_mask.sum()),
                "action_mask": self.action_mask.copy(),
                "center_occupied": bool(self.board[3, 3] == 1.0),
                "center_win": bool(pegs_remaining == 1 and self.board[3, 3] == 1.0),
            }

            return observation, reward, terminated, False, info

        def render(self):
            """Imprime el tablero en formato ASCII"""
            print("\n    0 1 2 3 4 5 6")
            for i, row in enumerate(self.board):
                row_str = f" {i}  "
                for val in row:
                    if val == 0.0:
                        row_str += ". "
                    elif val == 1.0:
                        row_str += "O "
                    else:
                        row_str += "  "
                print(row_str)
            print(f"\nFichas restantes: {int(np.sum(self.board))}")
            print(f"Movimientos disponibles: {len(self.valid_moves)}\n")


    # Crear instancia del entorno
    env = SimplifiedFrenchSolitaireEnv()
    obs, info = env.reset(seed=42)

    mo.md(f"""
    ### Entorno implementado ✅

    **Espacios definidos:**
    - Observación: {env.observation_space}
    - Acción: {env.action_space}

    **Estado inicial:**
    - Fichas: {info["pegs_remaining"]}
    - Forma de observación: {obs.shape}
    - Valores únicos: {np.unique(obs)}
    """)
    return SimplifiedFrenchSolitaireEnv, env


@app.cell(hide_code=True)
def _(env, mo, np):
    # Simular un paso aleatorio
    obs0, info0 = env.reset(seed=42)
    # Elegir la primera acción válida usando la máscara
    valid_indices = np.flatnonzero(info0.get("action_mask", []))
    if len(valid_indices) > 0:
        action = int(valid_indices[0])
    else:
        action = 0
    obs_new, reward, terminated, truncated, info_step = env.step(action)

    mo.md(f"""
    ### Ejemplo de interacción con el entorno

    ```python
    env.reset()
    action = next(iter(np.flatnonzero(info['action_mask'])))  # Índice de una acción válida
    obs, reward, terminated, truncated, info = env.step(action)
    ```

    **Resultado:**
    - Recompensa: `{reward}`
    - Terminado: `{terminated}`
    - Fichas restantes: `{info_step["pegs_remaining"]}`
    - Movimiento válido: `{info_step["valid"]}`

    ✅ El entorno funciona correctamente!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🧠 Parte 3: Algoritmo DQN (Deep Q-Network)

    ### ¿Qué es DQN?

    DQN es un algoritmo de RL que combina:
    - **Q-Learning**: método clásico de RL para aprender valores $Q(s, a)$
    - **Deep Learning**: redes neuronales para aproximar la función $Q$
    - **Experience Replay**: buffer de experiencias pasadas para estabilizar el entrenamiento
    - **Target Network**: red secundaria para cálculo de objetivos (reduce inestabilidad)

    ### Arquitectura de la red neuronal

    Para French Solitaire, una red simple funciona bien:

    ```
    Input (49) → FC(128) → ReLU → FC(128) → ReLU → FC(100) → Q-values
    ```

    - **Input**: estado del tablero (49 valores)
    - **Output**: Q-value para cada acción posible (100 acciones)

    ### Proceso de entrenamiento

    1. **Exploración**: el agente toma acciones aleatorias ($\epsilon$-greedy)
    2. **Almacenamiento**: guardar $(s, a, r, s', \text{done})$ en replay buffer
    3. **Muestreo**: tomar batch aleatorio del buffer
    4. **Actualización**: minimizar pérdida entre $Q$ predicho y $Q$ objetivo
    5. **Actualización de target**: copiar pesos de red principal cada N pasos

    ### Ecuación de Bellman (objetivo de $Q$-learning)

    $$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

    Donde:
    - $r$: recompensa inmediata
    - $\gamma$: factor de descuento (0.95-0.99)
    - $s'$: siguiente estado
    - $\max_{a'} Q(s', a')$: mejor Q-value en el siguiente estado
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import random

    # Verificar GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class QNetwork(nn.Module):
        """Red neuronal para aproximar Q(s,a)"""

        def __init__(self, state_dim=49, action_dim=100, hidden_dim=128):
            super(QNetwork, self).__init__()

            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, state):
            """Forward pass: estado → Q-values"""
            return self.network(state)


    class ReplayBuffer:
        """Buffer de experiencias para Experience Replay"""

        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done, next_action_mask):
            """Añadir experiencia al buffer, incluyendo máscara de acciones válidas en el siguiente estado"""
            self.buffer.append(
                (state, action, reward, next_state, done, next_action_mask)
            )

        def sample(self, batch_size):
            """Muestrear batch aleatorio"""
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones, next_action_masks = zip(
                *batch
            )

            return (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                np.array(next_action_masks),
            )

        def __len__(self):
            return len(self.buffer)


    class DQNAgent:
        """Agente DQN para French Solitaire"""

        def __init__(
            self,
            state_dim=49,
            action_dim=100,
            lr=5e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=100,
        ):
            self.device = device
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq

            # Redes: principal y target
            self.q_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            # Optimizador
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

            # Replay buffer
            self.replay_buffer = ReplayBuffer(buffer_size)

            # Contador de actualizaciones
            self.update_count = 0

        def select_action(
            self, state, valid_moves_count=None, action_mask=None, training=True
        ):
            """
            Selecciona acción usando ε-greedy:
            - Con probabilidad ε: acción aleatoria (exploración)
            - Con probabilidad 1-ε: mejor acción según Q-network (explotación)
            """
            # Determinar máscara válida
            if action_mask is not None:
                valid_indices = np.flatnonzero(action_mask)
            else:
                # Fallback a conteo (asume primeras 'valid_moves_count' acciones válidas)
                if valid_moves_count is None:
                    valid_indices = np.arange(self.action_dim)
                else:
                    valid_indices = np.arange(valid_moves_count)

            if training and random.random() < self.epsilon:
                # Exploración: acción aleatoria entre las válidas
                return int(np.random.choice(valid_indices))
            else:
                # Explotación: mejor Q-value entre las válidas
                with torch.no_grad():
                    state_tensor = (
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    q_values = self.q_network(state_tensor).cpu().numpy()[0]
                    masked_q = np.full_like(q_values, -np.inf)
                    masked_q[valid_indices] = q_values[valid_indices]
                    return int(np.argmax(masked_q))

        def train_step(self):
            """Realiza un paso de entrenamiento usando un batch del replay buffer"""
            if len(self.replay_buffer) < self.batch_size:
                return None

            # Muestrear batch
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                next_action_masks,
            ) = self.replay_buffer.sample(self.batch_size)

            # Convertir a tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            next_action_masks = torch.BoolTensor(next_action_masks).to(self.device)

            # Q-values actuales: Q(s, a)
            current_q = (
                self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )

            # Double DQN con enmascarado de acciones inválidas en s'
            with torch.no_grad():
                # Máscara booleana: True donde acción es válida
                valid_mask = next_action_masks

                q_next_main = self.q_network(next_states)
                q_next_main = q_next_main.masked_fill(~valid_mask, -1e9)
                next_actions = q_next_main.argmax(dim=1)

                q_next_target = self.target_network(next_states)
                q_next_target = q_next_target.masked_fill(~valid_mask, -1e9)
                next_q = q_next_target.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)

                target_q = rewards + (1 - dones) * self.gamma * next_q

            # Pérdida (Huber)
            loss = nn.functional.smooth_l1_loss(current_q, target_q)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Actualizar target network periódicamente
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            return loss.item()

        def decay_epsilon(self):
            """Reduce epsilon para menos exploración con el tiempo"""
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    # Crear agente
    agent = DQNAgent()

    mo.md(f"""
    ### Componentes de DQN implementados ✅

    **1. Q-Network:**
    ```
    {agent.q_network}
    ```

    **2. Hiperparámetros:**
    - Learning rate: `{agent.optimizer.param_groups[0]["lr"]}`
    - Gamma (descuento): `{agent.gamma}`
    - Epsilon inicial: `{1.0}` → final: `{agent.epsilon_end}`
    - Batch size: `{agent.batch_size}`
    - Buffer capacity: `{10000}`

    **3. Dispositivo de cómputo:** `{device}`

    **Parámetros totales:** {sum(p.numel() for p in agent.q_network.parameters())}
    """)
    return agent, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🏋️ Parte 4: Loop de entrenamiento

    ### Algoritmo de entrenamiento DQN

    ```python
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. Seleccionar acción (ε-greedy)
            action = agent.select_action(state, num_valid_moves)

            # 2. Ejecutar acción en el entorno
            next_state, reward, done, info = env.step(action)

            # 3. Guardar experiencia en replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 4. Entrenar con batch del buffer
            loss = agent.train_step()

            state = next_state
            total_reward += reward

        # 5. Decay epsilon
        agent.decay_epsilon()

        # 6. Logging (MLflow)
        log_metrics(episode, total_reward, loss, epsilon)
    ```

    ### Métricas importantes

    - **Recompensa acumulada por episodio**: indicador de progreso
    - **Epsilon**: nivel de exploración actual
    - **Pérdida (loss)**: convergencia del Q-learning
    - **Tasa de victoria**: % de episodios ganados
    - **Fichas promedio al final**: qué tan cerca estuvo de ganar

    ### MLflow para tracking

    ```python
    import mlflow

    with mlflow.start_run():
        mlflow.log_params({
            "algorithm": "DQN",
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "batch_size": 64
        })

        for episode in range(episodes):
            # ... entrenamiento ...
            mlflow.log_metrics({
                "reward": total_reward,
                "epsilon": agent.epsilon,
                "loss": loss
            }, step=episode)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(SimplifiedFrenchSolitaireEnv, agent, mo, np):
    # Función de entrenamiento (versión extendida con logging y checkpoints)
    def train_dqn_demo(
        num_episodes=100,
        verbose=True,
        use_mlflow=True,
        run_name="tutorial-dqn",
        checkpoint_path="checkpoints/dqn_tutorial.pt",
        mlflow=None,
    ):
        """
        Entrena el agente DQN en French Solitaire.

        Parámetros:
        - num_episodes: episodios a entrenar
        - verbose: imprime promedios cada 10 episodios
        - use_mlflow: si True y mlflow disponible, registra parámetros y métricas
        - run_name: nombre de la corrida en MLflow
        - checkpoint_path: ruta donde guardar el checkpoint al final
        - mlflow: módulo mlflow ya importado (o None)
        """
        import os
        import torch

        env_train = SimplifiedFrenchSolitaireEnv()

        episode_rewards = []
        episode_losses = []
        episode_pegs = []
        wins = 0
        run_id = None

        # Extraer LR del optimizador
        try:
            lr = agent.optimizer.param_groups[0]["lr"]
        except Exception:
            lr = None

        # MLflow: iniciar corrida si aplica (tipo seguro)
        from typing import Any

        mlf: Any = mlflow
        active_mlflow = bool(
            use_mlflow and (mlf is not None) and hasattr(mlf, "start_run")
        )
        if active_mlflow:
            try:
                with mlf.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id
                    # Parámetros
                    mlf.log_params(
                        {
                            "algorithm": "DQN",
                            "learning_rate": lr if lr is not None else 1e-3,
                            "gamma": agent.gamma,
                            "batch_size": agent.batch_size,
                            "buffer_size": len(agent.replay_buffer.buffer)
                            if hasattr(agent.replay_buffer, "buffer")
                            else 10000,
                            "target_update_freq": agent.target_update_freq,
                            "action_dim": agent.action_dim,
                        }
                    )

                    # Loop de entrenamiento (con máscara de acciones)
                    max_steps = 200
                    for episode in range(num_episodes):
                        state, info = env_train.reset()
                        mask = info.get("action_mask")
                        total_reward = 0
                        done = False
                        losses = []
                        steps = 0

                        while (not done) and (steps < max_steps):
                            if mask is None or int(np.sum(mask)) == 0:
                                break

                            action = agent.select_action(
                                state, action_mask=mask, training=True
                            )

                            next_state, reward, done, truncated, info = (
                                env_train.step(action)
                            )

                            next_mask = info.get("action_mask")

                            agent.replay_buffer.push(
                                state,
                                action,
                                reward,
                                next_state,
                                done,
                                next_mask
                                if next_mask is not None
                                else np.zeros(agent.action_dim, dtype=np.int8),
                            )

                            loss = agent.train_step()
                            if loss is not None:
                                losses.append(loss)

                            state = next_state
                            mask = next_mask
                            total_reward += reward
                            steps += 1

                        # Decay epsilon por episodio
                        agent.decay_epsilon()

                        # Estadísticas por episodio
                        episode_rewards.append(total_reward)
                        episode_losses.append(np.mean(losses) if losses else 0)
                        episode_pegs.append(info["pegs_remaining"])
                        if info.get("center_win", False):
                            wins += 1

                        # Logging episodios
                        mlf.log_metrics(
                            {
                                "reward": float(total_reward),
                                "epsilon": float(agent.epsilon),
                                "loss": float(episode_losses[-1]),
                                "pegs": int(episode_pegs[-1]),
                                "wins_cumulative": int(wins),
                                "steps": int(steps),
                            },
                            step=episode,
                        )

                        if verbose and (episode + 1) % 10 == 0:
                            avg_reward = np.mean(episode_rewards[-10:])
                            print(
                                f"Ep {episode + 1}/{num_episodes} | "
                                f"Avg Reward: {avg_reward:.2f} | "
                                f"Epsilon: {agent.epsilon:.3f} | "
                                f"Wins: {wins}"
                            )

                    # Guardar checkpoint y registrarlo como artefacto
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(
                        {
                            "q_network": agent.q_network.state_dict(),
                            "optimizer": agent.optimizer.state_dict(),
                            "epsilon": agent.epsilon,
                            "episodes": num_episodes,
                        },
                        checkpoint_path,
                    )
                    if hasattr(mlf, "log_artifact"):
                        mlf.log_artifact(checkpoint_path)

                    # Métricas finales
                    mlf.log_metrics(
                        {
                            "win_rate": wins / num_episodes,
                            "avg_reward": float(
                                np.mean(episode_rewards)
                                if episode_rewards
                                else 0.0
                            ),
                        }
                    )

            except Exception as e:
                # Si falla MLflow, continuar sin logging
                print(f"[MLflow] Advertencia: {e}. Continuando sin logging.")
                active_mlflow = False

        if not active_mlflow:
            # Loop sin MLflow (idéntico, sin llamadas de logging)
            max_steps = 200
            for episode in range(num_episodes):
                state, info = env_train.reset()
                mask = info.get("action_mask")
                total_reward = 0
                done = False
                losses = []
                steps = 0

                while (not done) and (steps < max_steps):
                    if mask is None or int(np.sum(mask)) == 0:
                        break

                    action = agent.select_action(
                        state, action_mask=mask, training=True
                    )
                    next_state, reward, done, truncated, info = env_train.step(
                        action
                    )

                    next_mask = info.get("action_mask")
                    agent.replay_buffer.push(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        next_mask
                        if next_mask is not None
                        else np.zeros(agent.action_dim, dtype=np.int8),
                    )
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)

                    state = next_state
                    mask = next_mask
                    total_reward += reward
                    steps += 1

                agent.decay_epsilon()
                episode_rewards.append(total_reward)
                episode_losses.append(np.mean(losses) if losses else 0)
                episode_pegs.append(info["pegs_remaining"])
                if info.get("center_win", False):
                    wins += 1

                if verbose and (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(
                        f"Ep {episode + 1}/{num_episodes} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Epsilon: {agent.epsilon:.3f} | "
                        f"Wins: {wins}"
                    )

            # Guardar checkpoint al final (sin registrar artefacto)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "q_network": agent.q_network.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon,
                    "episodes": num_episodes,
                },
                checkpoint_path,
            )

        return {
            "rewards": episode_rewards,
            "losses": episode_losses,
            "pegs": episode_pegs,
            "wins": wins,
            "win_rate": wins / num_episodes,
            "checkpoint_path": checkpoint_path,
            "mlflow_run_id": run_id,
        }


    mo.md("""
    ### Función de entrenamiento definida ✅ (con logging y checkpoints)

    - Registra parámetros y métricas en **MLflow** (si está disponible)
    - Guarda un **checkpoint** en `checkpoints/dqn_tutorial.pt`

    Puedes re-ejecutarla con distintos episodios y nombres de corrida.
    """)
    return (train_dqn_demo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📊 Parte 5: Visualización y análisis

    ### Curvas de aprendizaje

    Después del entrenamiento, analiza:

    1. **Recompensa vs Episodios**: ¿aumenta con el tiempo?
    2. **Pérdida vs Episodios**: ¿converge?
    3. **Epsilon decay**: ¿disminuye gradualmente?
    4. **Tasa de victoria**: ¿mejora en episodios tardíos?

    ### Código para visualización

    ```python
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Recompensa
    axes[0, 0].plot(results['rewards'])
    axes[0, 0].set_title('Recompensa por Episodio')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Recompensa Total')

    # Pérdida
    axes[0, 1].plot(results['losses'])
    axes[0, 1].set_title('Pérdida (Loss)')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('MSE Loss')

    # Fichas restantes
    axes[1, 0].plot(results['pegs'])
    axes[1, 0].set_title('Fichas Restantes al Final')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Fichas')
    axes[1, 0].axhline(y=1, color='r', linestyle='--', label='Objetivo')
    axes[1, 0].legend()

    # Tasa de victoria acumulada
    wins_cumulative = np.cumsum([1 if p == 1 else 0 for p in results['pegs']])
    episodes_range = np.arange(1, len(wins_cumulative) + 1)
    win_rate_cumulative = wins_cumulative / episodes_range
    axes[1, 1].plot(win_rate_cumulative)
    axes[1, 1].set_title('Tasa de Victoria Acumulada')
    axes[1, 1].set_xlabel('Episodio')
    axes[1, 1].set_ylabel('Win Rate')

    plt.tight_layout()
    plt.show()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mlflow, mlflow_available, mo, train_dqn_demo):
    from datetime import datetime

    # Entrenamiento corto automático para el tutorial (rápido y con logging)
    EPISODES_TUTORIAL = 15_000
    RUN_NAME = f"tutorial-dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    results = train_dqn_demo(
        num_episodes=EPISODES_TUTORIAL,
        verbose=True,
        use_mlflow=mlflow_available,
        run_name=RUN_NAME,
        mlflow=mlflow,
    )

    run_info = (
        f"Run ID: `{results['mlflow_run_id']}` en experimento 'french-solitaire'"
        if (mlflow_available and results.get("mlflow_run_id"))
        else "Sin MLflow (no disponible en el entorno)"
    )

    mo.md(f"""
    ## 🚀 Entrenamiento automático del tutorial

    - Episodios: **{EPISODES_TUTORIAL}**
    - Win rate: **{results["win_rate"]:.2%}**
    - Checkpoint guardado en: `{results["checkpoint_path"]}`
    - {run_info}

    Para explorar los experimentos localmente con MLflow UI (opcional):

    ```powershell
    conda activate french-solitaire
    mlflow ui --backend-store-uri file:./mlruns
    ```
    """)
    return EPISODES_TUTORIAL, results


@app.cell(hide_code=True)
def _(np, results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Recompensa
    axes[0, 0].plot(results['rewards'])
    axes[0, 0].set_title('Recompensa por Episodio')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Recompensa Total')

    # Pérdida
    axes[0, 1].plot(results['losses'])
    axes[0, 1].set_title('Pérdida (Loss)')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('MSE Loss')

    # Fichas restantes
    axes[1, 0].plot(results['pegs'])
    axes[1, 0].set_title('Fichas Restantes al Final')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Fichas')
    axes[1, 0].axhline(y=1, color='r', linestyle='--', label='Objetivo')
    axes[1, 0].legend()

    # Tasa de victoria acumulada
    wins_cumulative = np.cumsum([1 if p == 1 else 0 for p in results['pegs']])
    episodes_range = np.arange(1, len(wins_cumulative) + 1)
    win_rate_cumulative = wins_cumulative / episodes_range
    axes[1, 1].plot(win_rate_cumulative)
    axes[1, 1].set_title('Tasa de Victoria Acumulada')
    axes[1, 1].set_xlabel('Episodio')
    axes[1, 1].set_ylabel('Win Rate')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🎮 Parte 6: Evaluación del agente entrenado

    ### Modo evaluación (sin exploración)

    Una vez entrenado, evalúa el agente con $\epsilon=0$ (greedy):

    ```python
    def evaluate_agent(agent, env, num_episodes=100):
        agent.epsilon = 0.0  # Sin exploración
        wins = 0
        avg_pegs = []

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            mask = info.get("action_mask")

            while not done:
                if mask is None or int(np.sum(mask)) == 0:
                    break
                action = agent.select_action(state, action_mask=mask, training=False)
                state, reward, done, _, info = env.step(action)
                mask = info.get("action_mask")

            avg_pegs.append(info["pegs_remaining"])
            if info.get("center_win", False):
                wins += 1

        return {
            "win_rate": wins / num_episodes,
            "avg_pegs": np.mean(avg_pegs)
        }

    eval_results = evaluate_agent(agent, env, num_episodes=100)
    print(f"Tasa de victoria: {eval_results['win_rate']:.2%}")
    print(f"Promedio de fichas restantes: {eval_results['avg_pegs']:.2f}")
    ```

    ### Guardar modelo

    ```python
    import torch

    # Guardar checkpoint
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode': episode
                },
       'checkpoints/dqn_french_solitaire.pt'
    )

    # Cargar checkpoint
    checkpoint = torch.load('checkpoints/dqn_french_solitaire.pt')
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.epsilon = checkpoint['epsilon']
    ```
    """)
    return


@app.cell(hide_code=True)
def _(agent, env, np):
    def evaluate_agent(agent, env, num_episodes=100):
        agent.epsilon = 0.0  # Sin exploración
        wins = 0
        avg_pegs = []

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            mask = info.get("action_mask")

            while not done:
                if mask is None or int(np.sum(mask)) == 0:
                    break
                action = agent.select_action(state, action_mask=mask, training=False)
                state, reward, done, _, info = env.step(action)
                mask = info.get("action_mask")

            avg_pegs.append(info["pegs_remaining"])
            if info.get("center_win", False):
                wins += 1

        return {
            "win_rate": wins / num_episodes,
            "avg_pegs": np.mean(avg_pegs)
        }

    eval_results = evaluate_agent(agent, env, num_episodes=100)
    print(f"Tasa de victoria: {eval_results['win_rate']:.2%}")
    print(f"Promedio de fichas restantes: {eval_results['avg_pegs']:.2f}")
    return


@app.cell(hide_code=True)
def _(EPISODES_TUTORIAL, agent, torch):
    # Guardar checkpoint
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode': EPISODES_TUTORIAL
        },
       'checkpoints/dqn_french_solitaire.pt'
    )

    # Cargar checkpoint
    checkpoint = torch.load('checkpoints/dqn_french_solitaire.pt', weights_only=True)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.epsilon = checkpoint['epsilon']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🚀 Parte 7: Próximos pasos y mejoras

    ### Optimizaciones del algoritmo

    1. **Double DQN**: reduce overestimation de Q-values
       ```python
       # En train_step(), cambiar:
       next_action = self.q_network(next_states).argmax(1)
       next_q = self.target_network(next_states).gather(1, next_action.unsqueeze(1))
       ```

    2. **Dueling DQN**: separar value y advantage streams
       ```python
       self.value_stream = nn.Linear(hidden_dim, 1)
       self.advantage_stream = nn.Linear(hidden_dim, action_dim)
       ```

    3. **Prioritized Experience Replay**: muestrear experiencias importantes
       - Asignar prioridad basada en TD-error
       - Actualizar prioridades tras cada batch

    4. **Noisy Networks**: reemplazar ε-greedy con ruido en pesos

    ### Algoritmos alternativos

    - **PPO** (Proximal Policy Optimization): mejor para espacios de acción complejos
    - **A2C** (Advantage Actor-Critic): entrena policy y value simultáneamente
    - **Rainbow DQN**: combina 6 mejoras de DQN

    ### Ingeniería de características

    - **Simetría del tablero**: data augmentation con rotaciones/reflejos
    - **Features adicionales**: número de fichas, conectividad del tablero
    - **Curriculum learning**: empezar con estados cercanos a la victoria

    ### Producción

    1. Implementar entorno completo en `envs/french_solitaire_env.py`
    2. Separar código de agente en `agent/dqn.py`
    3. Script de entrenamiento con argparse en `train.py`
    4. Tests unitarios en `tests/`
    5. CI/CD con GitHub Actions
    6. MLflow para tracking en servidor remoto

    ### Recursos adicionales

    - **Paper original DQN**: [Playing Atari with Deep RL (2013)](https://arxiv.org/abs/1312.5602)
    - **Spinning Up in Deep RL**: https://spinningup.openai.com/
    - **Stable-Baselines3**: implementaciones de referencia (para comparar, no copiar)
    - **Gymnasium docs**: https://gymnasium.farama.org/
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📝 Resumen y checklist

    ### ✅ Lo que aprendiste

    - [x] Conceptos fundamentales de RL (estado, acción, recompensa, política)
    - [x] Implementación de entorno Gymnasium para French Solitaire
    - [x] Arquitectura de DQN con PyTorch
    - [x] Experience Replay y Target Network
    - [x] Loop de entrenamiento completo
    - [x] Tracking de experimentos con MLflow
    - [x] Evaluación y guardado de modelos

    ### 🛠️ Para implementar en producción

    1. **Estructura del proyecto**
       ```
       french-solitaire/
       ├── envs/
       │   └── french_solitaire_env.py  ← Implementar versión completa
       ├── agent/
       │   ├── dqn.py                   ← Clase DQNAgent
       │   ├── networks.py              ← QNetwork, Dueling, etc.
       │   └── replay_buffer.py         ← ReplayBuffer con priorización
       ├── scripts/
       │   └── train_dqn.py             ← Script de entrenamiento CLI
       ├── tests/
       │   ├── test_env.py              ← Tests del entorno
       │   └── test_agent.py            ← Tests del agente
       ├── checkpoints/                 ← Modelos guardados (.pt)
       ├── mlruns/                      ← Experimentos MLflow
       ├── train.py                     ← Entrypoint principal
       ├── eval.py                      ← Evaluación de modelos
       └── environment.yml              ← Ya creado ✅
       ```

    2. **Comandos de desarrollo**
       ```powershell
       # Activar entorno
       conda activate french-solitaire

       # Entrenar DQN
       python train.py --algo dqn --episodes 10000 --lr 1e-3 --gamma 0.99

       # Evaluar modelo
       python eval.py --checkpoint checkpoints/dqn_ep10000.pt --episodes 100

       # MLflow UI
       mlflow ui

       # Tests
       pytest tests/ -v
       ```

    3. **Hiperparámetros a tunear**
       - Learning rate: [1e-4, 1e-3, 1e-2]
       - Gamma: [0.95, 0.99, 0.999]
       - Batch size: [32, 64, 128]
       - Buffer size: [10k, 50k, 100k]
       - Hidden dim: [64, 128, 256]
       - Epsilon decay: [0.99, 0.995, 0.999]

    ### 🎯 Objetivo final

    **Entrenar un agente DQN que resuelva French Solitaire con >80% de tasa de victoria**

    Baseline esperado:
    - Random agent: ~0% victoria
    - Greedy heuristic: ~15% victoria
    - DQN bien entrenado: 50-80% victoria
    - DQN + mejoras (Double, Dueling, PER): 80-95% victoria

    ### 💡 Tips finales

    - **Empieza simple**: entrena con pocos episodios para validar el pipeline
    - **Log todo**: usa MLflow desde el inicio
    - **Checkpoints frecuentes**: guarda cada 1000 episodios
    - **Visualiza**: grafica curvas de aprendizaje regularmente
    - **Itera**: ajusta recompensas, arquitectura, hiperparámetros basado en resultados

    **¡Éxito con tu proyecto de RL! 🚀**
    """)
    return


@app.cell(hide_code=True)
def _(np, torch):
    import time

    def _dir_to_str(dr, dc):
        if dr == 0 and dc == 1:
            return "→"
        if dr == 0 and dc == -1:
            return "←"
        if dr == 1 and dc == 0:
            return "↓"
        if dr == -1 and dc == 0:
            return "↑"
        return f"({dr},{dc})"

    def describe_action(env, action_idx):
        """
        Devuelve una descripción legible para una acción (índice → (r_from,c_from)->(r_to,c_to), dirección)
        """
        if action_idx < 0 or action_idx >= len(env.all_actions):
            return f"acción {action_idx} (fuera de rango)"
        r_from, c_from, dr, dc = env.all_actions[action_idx]
        r_to, c_to = r_from + 2*dr, c_from + 2*dc
        return f"{(r_from, c_from)} -> {(r_to, c_to)} dir={_dir_to_str(dr, dc)}"

    def render_board_ascii(board):
        """
        Dibuja el tablero 7x7 como ASCII (igual a env.render pero sin espacios de celdas inválidas).
        """
        print("\n    0 1 2 3 4 5 6")
        for i, row in enumerate(board):
            row_str = f" {i}  "
            for val in row:
                if val == 1.0:
                    row_str += "O "
                elif val == 0.0:
                    row_str += ". "
                else:
                    row_str += "  "
            print(row_str)

    def run_greedy_episode(agent, env, top_k=5, pause=False, delay=0.0, max_steps=200, render_each_step=True):
        """
        Ejecuta un episodio con ε=0, mostrando paso a paso:
        - Top-K acciones por Q-value (enmascaradas por la máscara de acciones válidas)
        - Acción elegida, recompensa y fichas restantes
        - Tablero tras cada movimiento

        Parámetros:
          - agent: DQNAgent ya entrenado
          - env: instancia del entorno (p. ej., SimplifiedFrenchSolitaireEnv())
          - top_k: cuántas mejores acciones mostrar por paso
          - pause: si True, espera Enter en cada paso
          - delay: segundos de espera entre pasos (si pause=False)
          - max_steps: límite de pasos por episodio
          - render_each_step: si True, imprime el tablero después de cada acción

        Retorna:
          dict con la trayectoria (states, actions, rewards, pegs, done, steps)
        """
        prev_mode = agent.q_network.training
        agent.q_network.eval()
        try:
            traj = {
                "states": [],
                "actions": [],
                "rewards": [],
                "pegs": [],
                "done": None,
                "steps": 0
            }

            state, info = env.reset()
            mask = info.get("action_mask")
            pegs = info.get("pegs_remaining", None)

            print("=== INICIO DEL EPISODIO (modo greedy, ε=0) ===")
            print(f"Fichas iniciales: {pegs}, acciones válidas: {int(mask.sum()) if mask is not None else 0}")
            render_board_ascii(env.board)

            steps = 0
            done = False

            while not done and steps < max_steps:
                if mask is None or int(np.sum(mask)) == 0:
                    print("Sin acciones válidas. Episodio termina.")
                    break

                # Q-values y enmascarado
                with torch.no_grad():
                    s = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    q_all = agent.q_network(s).cpu().numpy()[0]

                valid_indices = np.flatnonzero(mask.astype(bool))
                q_masked = np.full_like(q_all, -np.inf, dtype=float)
                q_masked[valid_indices] = q_all[valid_indices]

                # Top-K acciones por Q
                k = min(top_k, len(valid_indices))
                top_idxs = valid_indices[np.argsort(q_all[valid_indices])[::-1][:k]]

                print(f"\n--- Paso {steps + 1} ---")
                print(f"Fichas: {int(np.sum(env.board))} | Acciones válidas: {len(valid_indices)}")

                for rank, idx in enumerate(top_idxs, start=1):
                    desc = describe_action(env, int(idx))
                    print(f"  Top {rank}: a={int(idx):3d} | Q={q_all[int(idx)]: .4f} | {desc}")

                # Acción elegida (greedy)
                action = int(np.argmax(q_masked))
                desc_best = describe_action(env, action)
                print(f"➡️  Acción elegida: a={action} | {desc_best}")

                # Ejecutar acción
                next_state, reward, done, truncated, info = env.step(action)
                next_mask = info.get("action_mask")
                pegs = info.get("pegs_remaining", None)

                # Post-acción
                if render_each_step:
                    render_board_ascii(env.board)
                print(f"Recompensa: {reward:+.1f} | Fichas restantes: {pegs} | Done: {done}")

                # Guardar en trayectoria
                traj["states"].append(state)
                traj["actions"].append(action)
                traj["rewards"].append(reward)
                traj["pegs"].append(pegs)

                # Avanzar
                state, mask = next_state, next_mask
                steps += 1

                if pause:
                    input("Presiona Enter para continuar...")
                elif delay and delay > 0:
                    time.sleep(delay)

            traj["done"] = done
            traj["steps"] = steps

            print("\n=== FIN DEL EPISODIO ===")
            outcome = "VICTORIA 🎉" if (done and pegs == 1) else "DERROTA ❌"
            print(f"Resultado: {outcome} | Pasos: {steps} | Fichas finales: {pegs}")
            return traj
        finally:
            # Restaurar modo de la red
            agent.q_network.train(prev_mode)
    return (run_greedy_episode,)


@app.cell(hide_code=True)
def _(SimplifiedFrenchSolitaireEnv, agent, run_greedy_episode):
    # Ejecuta un episodio greedy mostrando paso a paso
    # Usa una nueva instancia del entorno para no interferir con el que ya tengas
    demo_env = SimplifiedFrenchSolitaireEnv()
    traj = run_greedy_episode(
        agent,
        demo_env,
        top_k=5,        # muestra las 5 mejores acciones por Q
        pause=False,    # pon True si quieres avanzar con Enter
        delay=0.2,      # o un pequeño delay entre pasos
        max_steps=200,
        render_each_step=True
    )
    return


if __name__ == "__main__":
    app.run()
