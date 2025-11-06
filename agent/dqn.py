"""
Agente DQN para French Solitaire

Implementa Deep Q-Network con:
- Experience Replay
- Target Network
- Double DQN
- Epsilon-greedy exploration
- Soporte para action masking
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.networks import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Agente DQN para entrenamiento en French Solitaire.
    
    Características:
    - Double DQN para reducir overestimation
    - Action masking para acciones inválidas
    - Epsilon-greedy con decay exponencial
    - Target network para estabilidad
    
    Args:
        state_dim (int): dimensión del espacio de observación (default: 49)
        action_dim (int): dimensión del espacio de acción (default: 100)
        lr (float): learning rate (default: 5e-4)
        gamma (float): factor de descuento (default: 0.99)
        epsilon_start (float): epsilon inicial (default: 1.0)
        epsilon_end (float): epsilon mínimo (default: 0.01)
        epsilon_decay (float): tasa de decay de epsilon (default: 0.995)
        buffer_size (int): tamaño del replay buffer (default: 10000)
        batch_size (int): tamaño del batch (default: 64)
        target_update_freq (int): frecuencia de actualización de target network (default: 100)
        device (str): 'cuda' o 'cpu' (default: auto-detect)
    """
    
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
        device=None,
    ):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Hiperparámetros
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Redes: principal y target
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network siempre en modo eval
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Contador de actualizaciones
        self.update_count = 0
    
    def select_action(self, state, action_mask=None, training=True):
        """
        Selecciona acción usando ε-greedy con action masking.
        
        Args:
            state (np.ndarray): estado actual
            action_mask (np.ndarray): máscara de acciones válidas (1=válida, 0=inválida)
            training (bool): si está en modo entrenamiento (usa epsilon)
        
        Returns:
            int: índice de la acción seleccionada
        """
        # Determinar acciones válidas
        if action_mask is not None:
            valid_indices = np.flatnonzero(action_mask)
        else:
            valid_indices = np.arange(self.action_dim)
        
        if len(valid_indices) == 0:
            # No hay acciones válidas, retornar acción 0 (será penalizada)
            return 0
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # Exploración: acción aleatoria entre las válidas
            return int(np.random.choice(valid_indices))
        else:
            # Explotación: mejor Q-value entre las válidas
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
                
                # Enmascarar acciones inválidas
                if action_mask is not None:
                    q_values_masked = np.full_like(q_values, -np.inf)
                    q_values_masked[valid_indices] = q_values[valid_indices]
                    return int(np.argmax(q_values_masked))
                else:
                    return int(np.argmax(q_values))
    
    def train_step(self):
        """
        Realiza un paso de entrenamiento usando un batch del replay buffer.
        
        Implementa Double DQN:
        - Selecciona acción con Q-network principal
        - Evalúa con target network
        
        Returns:
            float: pérdida del batch (None si buffer insuficiente)
        """
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
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN con enmascarado de acciones inválidas
        with torch.no_grad():
            # Seleccionar acción con Q-network principal
            q_next_main = self.q_network(next_states)
            q_next_main = q_next_main.masked_fill(~next_action_masks, -1e9)
            next_actions = q_next_main.argmax(dim=1)
            
            # Evaluar con target network
            q_next_target = self.target_network(next_states)
            q_next_target = q_next_target.masked_fill(~next_action_masks, -1e9)
            next_q = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Target: r + γ * Q_target(s', a*)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Pérdida (Huber loss para robustez)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Actualizar target network periódicamente
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Reduce epsilon para menos exploración con el tiempo."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """
        Guarda el checkpoint del agente.
        
        Args:
            path (str): ruta del archivo .pt
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
            },
            path,
        )
    
    def load(self, path, load_optimizer=True):
        """
        Carga un checkpoint del agente.
        
        Args:
            path (str): ruta del archivo .pt
            load_optimizer (bool): si cargar el estado del optimizador
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        
        if "update_count" in checkpoint:
            self.update_count = checkpoint["update_count"]
