"""
Replay Buffers para DQN

Incluye:
- ReplayBuffer: buffer estándar con muestreo uniforme
- PrioritizedReplayBuffer: muestreo basado en TD-error (futuro)
"""
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Buffer de experiencias para Experience Replay en DQN.
    
    Almacena transiciones (state, action, reward, next_state, done, next_action_mask)
    y permite muestrear batches aleatorios para entrenamiento.
    
    Args:
        capacity (int): tamaño máximo del buffer (default: 10000)
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_action_mask):
        """
        Añade una experiencia al buffer.
        
        Args:
            state (np.ndarray): estado actual
            action (int): acción tomada
            reward (float): recompensa obtenida
            next_state (np.ndarray): siguiente estado
            done (bool): si el episodio terminó
            next_action_mask (np.ndarray): máscara de acciones válidas en next_state
        """
        self.buffer.append(
            (state, action, reward, next_state, done, next_action_mask)
        )
    
    def sample(self, batch_size):
        """
        Muestrea un batch aleatorio del buffer.
        
        Args:
            batch_size (int): tamaño del batch
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones, next_action_masks)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(next_action_masks),
        )
    
    def __len__(self):
        """Retorna el tamaño actual del buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Muestrea experiencias con probabilidad proporcional a su TD-error.
    Implementación futura para mejorar el entrenamiento.
    
    TODO: Implementar muestreo prioritizado con SumTree
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity (int): tamaño máximo del buffer
            alpha (float): qué tanto usar priorización (0=uniforme, 1=full priority)
            beta_start (float): compensación inicial de importance sampling
            beta_frames (int): frames para annealing de beta
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Por ahora, usar buffer estándar
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_action_mask, error=None):
        """
        Añade experiencia con prioridad.
        
        Args:
            error (float): TD-error inicial (si None, usar máxima prioridad)
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append((state, action, reward, next_state, done, next_action_mask))
        self.priorities.append(max_priority if error is None else error)
    
    def sample(self, batch_size):
        """
        Muestrea batch con priorización.
        
        TODO: Implementar muestreo proporcional a prioridades
        Por ahora, usa muestreo uniforme (equivalente a ReplayBuffer)
        """
        # Fallback a muestreo uniforme
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(next_action_masks),
        )
    
    def update_priorities(self, indices, errors):
        """
        Actualiza prioridades basadas en TD-errors.
        
        Args:
            indices (list): índices de las experiencias
            errors (list): nuevos TD-errors
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5  # epsilon para evitar prioridad cero
    
    def __len__(self):
        return len(self.buffer)
