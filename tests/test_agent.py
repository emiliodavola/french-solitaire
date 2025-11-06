"""
Tests para el agente DQN
"""
import pytest
import torch
import numpy as np

from agent.dqn import DQNAgent
from agent.networks import QNetwork, DuelingQNetwork
from agent.replay_buffer import ReplayBuffer


class TestQNetwork:
    """Tests para QNetwork."""
    
    def test_network_creation(self):
        """Test de creación de la red."""
        net = QNetwork(state_dim=49, action_dim=100, hidden_dim=128)
        assert net is not None
        
        # Verificar parámetros
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params > 0
    
    def test_forward_pass(self):
        """Test del forward pass."""
        net = QNetwork(state_dim=49, action_dim=100)
        state = torch.randn(32, 49)  # Batch de 32
        
        q_values = net(state)
        
        assert q_values.shape == (32, 100)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()
    
    def test_single_state(self):
        """Test con un solo estado."""
        net = QNetwork(state_dim=49, action_dim=100)
        state = torch.randn(1, 49)
        
        q_values = net(state)
        
        assert q_values.shape == (1, 100)


class TestDuelingQNetwork:
    """Tests para DuelingQNetwork."""
    
    def test_dueling_creation(self):
        """Test de creación de Dueling network."""
        net = DuelingQNetwork(state_dim=49, action_dim=100)
        assert net is not None
    
    def test_dueling_forward(self):
        """Test del forward pass en Dueling."""
        net = DuelingQNetwork(state_dim=49, action_dim=100)
        state = torch.randn(16, 49)
        
        q_values = net(state)
        
        assert q_values.shape == (16, 100)
        assert not torch.isnan(q_values).any()


class TestReplayBuffer:
    """Tests para ReplayBuffer."""
    
    def test_buffer_creation(self):
        """Test de creación del buffer."""
        buffer = ReplayBuffer(capacity=1000)
        assert len(buffer) == 0
    
    def test_push_and_sample(self):
        """Test de push y sample."""
        buffer = ReplayBuffer(capacity=100)
        
        # Añadir experiencias
        for i in range(50):
            state = np.random.rand(49)
            action = np.random.randint(0, 100)
            reward = np.random.rand()
            next_state = np.random.rand(49)
            done = False
            next_mask = np.random.randint(0, 2, size=100)
            
            buffer.push(state, action, reward, next_state, done, next_mask)
        
        assert len(buffer) == 50
        
        # Muestrear
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones, masks = batch
        
        assert states.shape == (32, 49)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 49)
        assert dones.shape == (32,)
        assert masks.shape == (32, 100)
    
    def test_buffer_overflow(self):
        """Test de overflow del buffer."""
        buffer = ReplayBuffer(capacity=10)
        
        # Añadir más experiencias que la capacidad
        for i in range(20):
            buffer.push(
                np.random.rand(49),
                i,
                0.0,
                np.random.rand(49),
                False,
                np.ones(100),
            )
        
        # Debe mantener solo las últimas 10
        assert len(buffer) == 10


class TestDQNAgent:
    """Tests para DQNAgent."""
    
    def test_agent_creation(self):
        """Test de creación del agente."""
        agent = DQNAgent(
            state_dim=49,
            action_dim=100,
            lr=1e-3,
            gamma=0.99,
            device="cpu",
        )
        
        assert agent is not None
        assert agent.device.type == "cpu"
        assert agent.gamma == 0.99
    
    def test_select_action_greedy(self):
        """Test de selección de acción greedy."""
        agent = DQNAgent(device="cpu")
        agent.epsilon = 0.0  # Greedy
        
        state = np.random.rand(49)
        action_mask = np.ones(100)
        
        action = agent.select_action(state, action_mask=action_mask, training=False)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 100
    
    def test_select_action_with_mask(self):
        """Test de selección de acción con máscara."""
        agent = DQNAgent(device="cpu")
        agent.epsilon = 0.0
        
        state = np.random.rand(49)
        
        # Máscara que permite solo 5 acciones
        action_mask = np.zeros(100)
        valid_indices = [10, 20, 30, 40, 50]
        action_mask[valid_indices] = 1
        
        action = agent.select_action(state, action_mask=action_mask, training=False)
        
        # La acción debe estar en las válidas
        assert action in valid_indices
    
    def test_train_step_insufficient_buffer(self):
        """Test de train_step con buffer insuficiente."""
        agent = DQNAgent(device="cpu", batch_size=64)
        
        # Añadir menos experiencias que batch_size
        for i in range(10):
            agent.replay_buffer.push(
                np.random.rand(49),
                0,
                1.0,
                np.random.rand(49),
                False,
                np.ones(100),
            )
        
        # No debe entrenar
        loss = agent.train_step()
        assert loss is None
    
    def test_train_step_with_sufficient_buffer(self):
        """Test de train_step con buffer suficiente."""
        agent = DQNAgent(device="cpu", batch_size=32, buffer_size=1000)
        
        # Llenar buffer
        for i in range(100):
            agent.replay_buffer.push(
                np.random.rand(49),
                np.random.randint(0, 100),
                np.random.rand(),
                np.random.rand(49),
                False,
                np.random.randint(0, 2, size=100),
            )
        
        # Entrenar
        loss = agent.train_step()
        
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0  # Pérdida no puede ser negativa
    
    def test_epsilon_decay(self):
        """Test de decay de epsilon."""
        agent = DQNAgent(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.99,
            device="cpu",
        )
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_end
        
        # Decay repetido
        for _ in range(1000):
            agent.decay_epsilon()
        
        # No debe bajar del mínimo
        assert agent.epsilon >= agent.epsilon_end
    
    def test_save_and_load(self, tmp_path):
        """Test de guardar y cargar checkpoint."""
        agent1 = DQNAgent(device="cpu")
        
        # Entrenar un poco para cambiar pesos
        for i in range(100):
            agent1.replay_buffer.push(
                np.random.rand(49),
                0,
                1.0,
                np.random.rand(49),
                False,
                np.ones(100),
            )
        
        for _ in range(10):
            agent1.train_step()
        
        # Guardar
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        agent1.save(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Cargar en nuevo agente
        agent2 = DQNAgent(device="cpu")
        agent2.load(str(checkpoint_path))
        
        # Verificar que epsilon se cargó
        assert agent2.epsilon == agent1.epsilon
        
        # Verificar que los pesos son iguales
        for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
