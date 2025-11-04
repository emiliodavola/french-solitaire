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
    mo.md("""
    ## 📚 Parte 1: Conceptos básicos de RL

    ### ¿Qué es Reinforcement Learning?

    El aprendizaje por refuerzo es un paradigma de ML donde un **agente** aprende a tomar
    **decisiones** en un **entorno** para maximizar una **recompensa acumulada**.

    #### Componentes clave:

    - **Entorno**: el juego French Solitaire
    - **Estado (s)**: configuración actual del tablero (32 posiciones)
    - **Acción (a)**: mover una ficha (origen → destino)
    - **Recompensa (r)**: feedback numérico tras cada acción
    - **Política (π)**: estrategia del agente para elegir acciones
    - **Valor Q(s,a)**: recompensa esperada al tomar acción `a` en estado `s`

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

            # Espacio de acción: discreto (usaremos índice de movimiento)
            # Simplificación: máximo ~76 movimientos posibles en cualquier estado
            self.action_space = spaces.Discrete(100)

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

            self.board = None
            self.valid_moves = []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            # Crear tablero inicial
            self.board = np.zeros((7, 7), dtype=np.float32)
            for r, c in self.valid_positions:
                self.board[r, c] = 1.0
            self.board[3, 3] = 0.0  # Centro vacío

            self._update_valid_moves()

            observation = self.board.flatten()
            info = {"pegs_remaining": int(np.sum(self.board))}

            return observation, info

        def _update_valid_moves(self):
            """Actualiza lista de movimientos válidos en el estado actual"""
            self.valid_moves = []
            directions = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
            ]  # derecha, izq, abajo, arriba

            for r, c in self.valid_positions:
                if self.board[r, c] == 1:  # Hay ficha
                    for dr, dc in directions:
                        jump_r, jump_c = r + dr, c + dc
                        land_r, land_c = r + 2 * dr, c + 2 * dc

                        # Validar movimiento
                        if (
                            0 <= land_r < 7
                            and 0 <= land_c < 7
                            and (land_r, land_c) in self.valid_positions
                            and self.board[jump_r, jump_c] == 1
                            and self.board[land_r, land_c] == 0
                        ):
                            self.valid_moves.append(((r, c), (land_r, land_c)))

        def step(self, action):
            """
            Ejecuta una acción (índice de movimiento)
            Retorna: observation, reward, terminated, truncated, info
            """
            # Validar que la acción sea válida
            if action >= len(self.valid_moves):
                # Acción inválida (el agente eligió un índice fuera de rango)
                reward = -10.0
                observation = self.board.flatten()
                info = {"pegs_remaining": int(np.sum(self.board)), "valid": False}
                return observation, reward, False, False, info

            # Ejecutar movimiento válido
            (r_from, c_from), (r_to, c_to) = self.valid_moves[action]
            dr = (r_to - r_from) // 2
            dc = (c_to - c_from) // 2
            r_jump, c_jump = r_from + dr, c_from + dc

            # Aplicar movimiento
            self.board[r_from, c_from] = 0.0
            self.board[r_jump, c_jump] = 0.0
            self.board[r_to, c_to] = 1.0

            pegs_remaining = int(np.sum(self.board))

            # Calcular recompensa
            if pegs_remaining == 1:
                reward = 100.0  # Victoria!
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
                "moves_available": len(self.valid_moves),
            }

            return observation, reward, terminated, False, info

        def render(self):
            """Imprime el tablero en formato ASCII"""
            symbols = {0.0: ".", 1.0: "O", -1.0: " "}
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
def _(env, mo):
    # Simular un paso aleatorio
    env.reset(seed=42)
    action = 0  # Primera acción válida
    obs_new, reward, terminated, truncated, info_step = env.step(action)

    mo.md(f"""
    ### Ejemplo de interacción con el entorno

    ```python
    env.reset()
    action = 0  # Primera acción de la lista de movimientos válidos
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

        def push(self, state, action, reward, next_state, done):
            """Añadir experiencia al buffer"""
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            """Muestrear batch aleatorio"""
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            return (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
            )

        def __len__(self):
            return len(self.buffer)


    class DQNAgent:
        """Agente DQN para French Solitaire"""

        def __init__(
            self,
            state_dim=49,
            action_dim=100,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=10,
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

        def select_action(self, state, valid_moves_count, training=True):
            """
            Selecciona acción usando ε-greedy:
            - Con probabilidad ε: acción aleatoria (exploración)
            - Con probabilidad 1-ε: mejor acción según Q-network (explotación)
            """
            if training and random.random() < self.epsilon:
                # Exploración: acción aleatoria entre las válidas
                return random.randint(0, valid_moves_count - 1)
            else:
                # Explotación: mejor Q-value
                with torch.no_grad():
                    state_tensor = (
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    q_values = self.q_network(state_tensor)

                    # Enmascarar acciones inválidas
                    q_values_np = q_values.cpu().numpy()[0]
                    q_values_np[valid_moves_count:] = -np.inf

                    return int(np.argmax(q_values_np))

        def train_step(self):
            """Realiza un paso de entrenamiento usando un batch del replay buffer"""
            if len(self.replay_buffer) < self.batch_size:
                return None

            # Muestrear batch
            states, actions, rewards, next_states, dones = (
                self.replay_buffer.sample(self.batch_size)
            )

            # Convertir a tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Q-values actuales: Q(s, a)
            current_q = (
                self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )

            # Q-values objetivo: r + γ * max_a' Q_target(s', a')
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q

            # Pérdida (MSE o Huber)
            loss = nn.functional.mse_loss(current_q, target_q)

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
    - Learning rate: `{1e-3}`
    - Gamma (descuento): `{agent.gamma}`
    - Epsilon inicial: `{1.0}` → final: `{agent.epsilon_end}`
    - Batch size: `{agent.batch_size}`
    - Buffer capacity: `{10000}`

    **3. Dispositivo de cómputo:** `{device}`

    **Parámetros totales:** {sum(p.numel() for p in agent.q_network.parameters())}
    """)
    return (agent,)


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
    # Función de entrenamiento (versión simplificada para demo)
    def train_dqn_demo(num_episodes=100, verbose=True):
        """Entrena el agente DQN en French Solitaire"""
        env_train = SimplifiedFrenchSolitaireEnv()

        episode_rewards = []
        episode_losses = []
        episode_pegs = []
        wins = 0

        for episode in range(num_episodes):
            state, info = env_train.reset()
            total_reward = 0
            done = False
            losses = []

            while not done:
                # Seleccionar acción
                valid_moves_count = len(env_train.valid_moves)
                if valid_moves_count == 0:
                    break

                action = agent.select_action(
                    state, valid_moves_count, training=True
                )

                # Ejecutar acción
                next_state, reward, done, truncated, info = env_train.step(action)

                # Guardar experiencia
                agent.replay_buffer.push(state, action, reward, next_state, done)

                # Entrenar
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward

            # Decay epsilon
            agent.decay_epsilon()

            # Estadísticas
            episode_rewards.append(total_reward)
            episode_losses.append(np.mean(losses) if losses else 0)
            episode_pegs.append(info["pegs_remaining"])

            if info["pegs_remaining"] == 1:
                wins += 1

            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(
                    f"Ep {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Wins: {wins}"
                )

        return {
            "rewards": episode_rewards,
            "losses": episode_losses,
            "pegs": episode_pegs,
            "wins": wins,
            "win_rate": wins / num_episodes,
        }


    # NOTA: No ejecutamos el entrenamiento automáticamente aquí
    # (consumiría mucho tiempo en una demo interactiva)
    # El usuario puede ejecutar manualmente: results = train_dqn_demo(100)

    mo.md("""
    ### Función de entrenamiento definida ✅

    Para entrenar el agente, ejecuta en una celda nueva:

    ```python
    results = train_dqn_demo(num_episodes=100, verbose=True)
    print(f"Tasa de victoria: {results['win_rate']:.2%}")
    ```

    **Advertencia**: El entrenamiento puede tardar varios minutos dependiendo de:
    - Número de episodios
    - Velocidad de GPU/CPU
    - Complejidad del entorno

    En producción, usarías:
    - 10,000+ episodios
    - Checkpointing cada 1000 episodios
    - Early stopping si converge
    - Tracking con MLflow
    """)
    return


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

            while not done:
                valid_moves = len(env.valid_moves)
                if valid_moves == 0:
                    break

                action = agent.select_action(state, valid_moves, training=False)
                state, reward, done, _, info = env.step(action)

            avg_pegs.append(info["pegs_remaining"])
            if info["pegs_remaining"] == 1:
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


if __name__ == "__main__":
    app.run()
