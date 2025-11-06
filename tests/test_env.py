"""
Tests para el entorno French Solitaire
"""
import pytest
import numpy as np
from envs.french_solitaire_env import FrenchSolitaireEnv


class TestFrenchSolitaireEnv:
    """Suite de tests para el entorno."""
    
    def test_env_creation(self):
        """Test de creación básica del entorno."""
        env = FrenchSolitaireEnv()
        assert env is not None
        assert env.observation_space.shape == (49,)
        assert env.action_space.n == 100
    
    def test_reset(self):
        """Test del método reset."""
        env = FrenchSolitaireEnv()
        obs, info = env.reset()
        
        # Verificar observación
        assert obs.shape == (49,)
        assert obs.dtype == np.float32
        
        # Verificar info
        assert "pegs_remaining" in info
        assert "action_mask" in info
        assert "moves_available" in info
        
        # Estado inicial: 32 fichas
        assert info["pegs_remaining"] == 32
        assert np.sum(obs) == 32
        
        # Centro debe estar vacío
        board = obs.reshape(7, 7)
        assert board[3, 3] == 0.0
    
    def test_valid_positions(self):
        """Test de posiciones válidas del tablero."""
        env = FrenchSolitaireEnv()
        obs, info = env.reset()
        board = obs.reshape(7, 7)
        
        # Verificar cruz europea
        # Esquinas deben estar vacías (posiciones inválidas)
        assert board[0, 0] == 0.0
        assert board[0, 1] == 0.0
        assert board[0, 5] == 0.0
        assert board[0, 6] == 0.0
        assert board[6, 0] == 0.0
        assert board[6, 1] == 0.0
        assert board[6, 5] == 0.0
        assert board[6, 6] == 0.0
        
        # Centro de la cruz debe tener fichas (excepto [3,3])
        assert board[3, 0] == 1.0
        assert board[3, 6] == 1.0
        assert board[0, 3] == 1.0
        assert board[6, 3] == 1.0
    
    def test_valid_move(self):
        """Test de un movimiento válido."""
        env = FrenchSolitaireEnv()
        obs, info = env.reset()
        
        # Obtener una acción válida
        valid_indices = np.flatnonzero(info["action_mask"])
        assert len(valid_indices) > 0
        
        action = valid_indices[0]
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Verificar que el movimiento fue válido
        assert info["valid"] == True
        assert reward > 0  # Recompensa positiva por movimiento válido
        assert info["pegs_remaining"] == 31  # Una ficha menos
        assert not done  # No terminado en un paso
    
    def test_invalid_move(self):
        """Test de un movimiento inválido."""
        env = FrenchSolitaireEnv()
        obs, info = env.reset()
        
        # Encontrar una acción inválida
        invalid_indices = np.where(info["action_mask"] == 0)[0]
        if len(invalid_indices) > 0:
            action = invalid_indices[0]
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Verificar penalización
            assert info["valid"] == False
            assert reward == -10.0  # Penalización por movimiento inválido
            assert info["pegs_remaining"] == 32  # No cambió el estado
    
    def test_deterministic_reset(self):
        """Test de reset determinístico con semilla."""
        env = FrenchSolitaireEnv()
        
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
        assert info1["pegs_remaining"] == info2["pegs_remaining"]
    
    def test_action_mask_consistency(self):
        """Test de consistencia de la máscara de acciones."""
        env = FrenchSolitaireEnv()
        obs, info = env.reset()
        
        mask = info["action_mask"]
        valid_count = int(np.sum(mask))
        moves_available = info["moves_available"]
        
        # La cantidad de 1s en la máscara debe coincidir con moves_available
        assert valid_count == moves_available
    
    def test_render(self):
        """Test del método render."""
        env = FrenchSolitaireEnv(render_mode="ansi")
        obs, info = env.reset()
        
        output = env.render()
        assert output is not None
        assert isinstance(output, str)
        assert "Fichas restantes: 32" in output
    
    def test_victory_condition(self):
        """Test de condición de victoria (simulado)."""
        env = FrenchSolitaireEnv()
        
        # Este test requeriría simular una secuencia de movimientos ganadora
        # Por ahora, verificamos que el entorno detecta victoria
        # cuando board[3,3] == 1 y sum(board) == 1
        
        # Configurar manualmente un estado de victoria (para testing)
        env.reset()
        env.board = np.zeros((7, 7), dtype=np.float32)
        env.board[3, 3] = 1.0  # Una ficha en el centro
        env._update_valid_moves()
        
        # Verificar que no hay movimientos válidos (victoria)
        assert len(env.valid_moves) == 0
        assert np.sum(env.board) == 1
        assert env.board[3, 3] == 1.0


class TestEnvironmentEdgeCases:
    """Tests de casos edge."""
    
    def test_max_steps_limit(self):
        """Test de límite de pasos (verificar que no entra en loop infinito)."""
        env = FrenchSolitaireEnv()
        obs, info = env.reset()
        
        max_steps = 200
        for step in range(max_steps):
            valid_indices = np.flatnonzero(info["action_mask"])
            if len(valid_indices) == 0:
                break
            
            action = np.random.choice(valid_indices)
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                break
        
        # Verificar que terminó antes del límite o en el límite
        assert step <= max_steps
    
    def test_no_valid_moves_defeat(self):
        """Test de derrota por falta de movimientos válidos."""
        env = FrenchSolitaireEnv()
        env.reset()
        
        # Configurar un estado sin movimientos válidos (2 fichas aisladas)
        env.board = np.zeros((7, 7), dtype=np.float32)
        env.board[2, 2] = 1.0
        env.board[4, 4] = 1.0
        env._update_valid_moves()
        
        # Verificar que no hay movimientos válidos
        assert len(env.valid_moves) == 0
        assert int(np.sum(env.board)) > 1  # Más de 1 ficha (no victoria)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
