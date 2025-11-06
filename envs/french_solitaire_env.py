"""
French Solitaire Environment (7x7 European variant)

Compatible con Gymnasium API para entrenamiento de RL.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FrenchSolitaireEnv(gym.Env):
    """
    Entorno de French Solitaire (Peg Solitaire 7x7) para Reinforcement Learning.
    
    Objetivo: Reducir 32 fichas a 1 sola ficha en el centro del tablero.
    
    Espacios:
        - observation_space: Box(49,) — tablero 7x7 aplanado
        - action_space: Discrete(100) — índices de acciones geométricamente válidas
    
    Recompensas:
        - +100: victoria (1 ficha en el centro)
        - +50: 1 ficha restante (pero no en centro)
        - +1: reducción de ficha
        - -10: movimiento inválido
        - -50: sin movimientos disponibles (derrota)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Espacio de observación: tablero 7x7 aplanado (49 elementos)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(49,), dtype=np.float32
        )
        
        # Direcciones de movimiento (derecha, izquierda, abajo, arriba)
        self.directions = [
            (0, 1),   # derecha
            (0, -1),  # izquierda
            (1, 0),   # abajo
            (-1, 0),  # arriba
        ]
        
        # Posiciones válidas en el tablero (cruz europea)
        self.valid_positions = set([
            (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
            (5, 2), (5, 3), (5, 4),
            (6, 2), (6, 3), (6, 4),
        ])
        
        # Precomputar todas las acciones geométricamente posibles
        self.all_actions = []
        for r, c in self.valid_positions:
            for dr, dc in self.directions:
                jump_r, jump_c = r + dr, c + dc
                land_r, land_c = r + 2 * dr, c + 2 * dc
                if (jump_r, jump_c) in self.valid_positions and (land_r, land_c) in self.valid_positions:
                    self.all_actions.append((r, c, dr, dc))
        
        # Espacio de acción: discreto con mapeo FIJO
        self.max_actions = 100
        if len(self.all_actions) > self.max_actions:
            self.all_actions = self.all_actions[:self.max_actions]
        self.action_space = spaces.Discrete(self.max_actions)
        
        # Estado inicial
        self.board = np.zeros((7, 7), dtype=np.float32)
        self.valid_moves = []
        self.action_mask = np.zeros(self.max_actions, dtype=np.int8)
    
    def reset(self, *, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            observation (np.ndarray): tablero aplanado (49,)
            info (dict): información adicional (pegs_remaining, action_mask)
        """
        super().reset(seed=seed)
        
        # Crear tablero inicial: todas las posiciones válidas con ficha excepto el centro
        self.board = np.zeros((7, 7), dtype=np.float32)
        for r, c in self.valid_positions:
            self.board[r, c] = 1.0
        self.board[3, 3] = 0.0  # Centro vacío
        
        self._update_valid_moves()
        
        observation = self.board.flatten()
        info = {
            "pegs_remaining": int(np.sum(self.board)),
            "action_mask": self.action_mask.copy(),
            "moves_available": len(self.valid_moves),
        }
        
        return observation, info
    
    def _update_valid_moves(self):
        """Actualiza la lista de movimientos válidos y la máscara de acciones."""
        self.valid_moves = []
        self.action_mask.fill(0)
        
        for idx, (r_from, c_from, dr, dc) in enumerate(self.all_actions):
            if idx >= self.max_actions:
                break
            
            jump_r, jump_c = r_from + dr, c_from + dc
            land_r, land_c = r_from + 2 * dr, c_from + 2 * dc
            
            # Validar: origen con ficha, salto con ficha, destino vacío
            if (
                self.board[r_from, c_from] == 1
                and self.board[jump_r, jump_c] == 1
                and self.board[land_r, land_c] == 0
            ):
                self.action_mask[idx] = 1
                self.valid_moves.append(((r_from, c_from), (land_r, land_c)))
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action (int): índice de la acción a ejecutar
        
        Returns:
            observation (np.ndarray): nuevo estado del tablero
            reward (float): recompensa obtenida
            terminated (bool): si el episodio terminó
            truncated (bool): si el episodio fue truncado (siempre False)
            info (dict): información adicional
        """
        # Validar acción
        if action >= self.max_actions or self.action_mask[action] == 0:
            # Acción inválida
            reward = -10.0
            observation = self.board.flatten()
            info = {
                "pegs_remaining": int(np.sum(self.board)),
                "valid": False,
                "action_mask": self.action_mask.copy(),
                "moves_available": len(self.valid_moves),
            }
            return observation, reward, False, False, info
        
        # Ejecutar movimiento válido
        r_from, c_from, dr, dc = self.all_actions[action]
        r_to, c_to = r_from + 2 * dr, c_from + 2 * dc
        r_jump, c_jump = r_from + dr, c_from + dc
        
        # Aplicar movimiento
        self.board[r_from, c_from] = 0.0
        self.board[r_jump, c_jump] = 0.0
        self.board[r_to, c_to] = 1.0
        
        pegs_remaining = int(np.sum(self.board))
        
        # Calcular recompensa
        if pegs_remaining == 1:
            if self.board[3, 3] == 1.0:
                reward = 100.0  # Victoria perfecta
            else:
                reward = 50.0   # 1 ficha pero no en centro
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
            "action_mask": self.action_mask.copy(),
            "center_occupied": bool(self.board[3, 3] == 1.0),
            "center_win": bool(pegs_remaining == 1 and self.board[3, 3] == 1.0),
        }
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """Renderiza el tablero en formato ASCII."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
        return None
    
    def _render_ansi(self):
        """Renderiza el tablero como string ASCII."""
        lines = ["\n    0 1 2 3 4 5 6"]
        for i, row in enumerate(self.board):
            row_str = f" {i}  "
            for val in row:
                if val == 1.0:
                    row_str += "O "
                elif val == 0.0:
                    row_str += ". "
                else:
                    row_str += "  "
            lines.append(row_str)
        lines.append(f"\nFichas restantes: {int(np.sum(self.board))}")
        lines.append(f"Movimientos disponibles: {len(self.valid_moves)}\n")
        
        output = "\n".join(lines)
        
        if self.render_mode == "human":
            print(output)
        
        return output
