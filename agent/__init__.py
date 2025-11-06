"""
Módulo de agentes de RL
"""
from agent.networks import QNetwork, DuelingQNetwork
from agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from agent.dqn import DQNAgent

__all__ = [
    "QNetwork",
    "DuelingQNetwork",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "DQNAgent",
]
