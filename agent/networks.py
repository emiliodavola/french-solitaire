"""
Redes neuronales para DQN

Incluye:
- QNetwork: red simple para aproximar Q(s,a)
- DuelingQNetwork: arquitectura Dueling DQN (separación value/advantage)
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Red neuronal para aproximar Q(s,a) en DQN.
    
    Arquitectura simple:
        Input (state_dim) → FC(hidden_dim) → ReLU → FC(hidden_dim) → ReLU → FC(action_dim)
    
    Args:
        state_dim (int): dimensión del espacio de observación (default: 49)
        action_dim (int): dimensión del espacio de acción (default: 100)
        hidden_dim (int): dimensión de capas ocultas (default: 128)
    """
    
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
        """
        Forward pass: estado → Q-values.
        
        Args:
            state (torch.Tensor): tensor de estados (batch_size, state_dim)
        
        Returns:
            torch.Tensor: Q-values para cada acción (batch_size, action_dim)
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Network: separa value stream y advantage stream.
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
    
    Ventajas:
    - Aprende qué estados son valiosos independientemente de las acciones
    - Mejora la convergencia en problemas donde muchas acciones tienen Q similar
    
    Args:
        state_dim (int): dimensión del espacio de observación
        action_dim (int): dimensión del espacio de acción
        hidden_dim (int): dimensión de capas ocultas
    """
    
    def __init__(self, state_dim=49, action_dim=100, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        
        # Feature extraction compartido
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        """
        Forward pass con arquitectura Dueling.
        
        Args:
            state (torch.Tensor): tensor de estados (batch_size, state_dim)
        
        Returns:
            torch.Tensor: Q-values combinados (batch_size, action_dim)
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        # Restar la media de advantages para identificabilidad
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
