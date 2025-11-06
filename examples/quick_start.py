"""
Script de ejemplo: Entrenar y evaluar un agente DQN en French Solitaire

Este script demuestra el flujo completo:
1. Crear entorno
2. Crear y entrenar agente DQN
3. Evaluar el agente entrenado
4. Guardar checkpoint

Uso:
    conda activate french-solitaire
    python examples/quick_start.py
"""
import sys
import os

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from envs.french_solitaire_env import FrenchSolitaireEnv
from agent.dqn import DQNAgent


def train_quick_demo(episodes=1000, verbose=True):
    """
    Entrenamiento rápido de demostración.
    
    Args:
        episodes (int): número de episodios
        verbose (bool): imprimir progreso
    
    Returns:
        DQNAgent: agente entrenado
    """
    # Crear entorno
    env = FrenchSolitaireEnv()
    
    # Crear agente
    agent = DQNAgent(
        state_dim=49,
        action_dim=100,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
    )
    
    print(f"=== Entrenamiento rápido: {episodes} episodios ===")
    print(f"Dispositivo: {agent.device}\n")
    
    # Métricas
    episode_rewards = []
    wins = 0
    
    # Loop de entrenamiento
    for episode in range(episodes):
        state, info = env.reset()
        mask = info.get("action_mask")
        total_reward = 0
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Seleccionar acción
            action = agent.select_action(state, action_mask=mask, training=True)
            
            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)
            next_mask = info.get("action_mask")
            
            # Guardar experiencia
            agent.replay_buffer.push(
                state, action, reward, next_state, done, next_mask
            )
            
            # Entrenar
            agent.train_step()
            
            # Avanzar
            state, mask = next_state, next_mask
            total_reward += reward
            steps += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Registrar métricas
        episode_rewards.append(total_reward)
        if info.get("center_win", False):
            wins += 1
        
        # Progreso
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            win_rate = wins / (episode + 1)
            print(f"Episodio {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"Win rate final: {wins / episodes:.2%}")
    
    return agent


def evaluate_agent(agent, env, num_episodes=100):
    """
    Evalúa el agente entrenado.
    
    Args:
        agent (DQNAgent): agente a evaluar
        env: entorno del juego
        num_episodes (int): número de episodios
    
    Returns:
        dict: resultados de evaluación
    """
    agent.epsilon = 0.0  # Sin exploración
    agent.q_network.eval()
    
    wins = 0
    center_wins = 0
    episode_rewards = []
    episode_pegs = []
    
    print(f"\n=== Evaluación: {num_episodes} episodios ===")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        mask = info.get("action_mask")
        total_reward = 0
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            action = agent.select_action(state, action_mask=mask, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            next_mask = info.get("action_mask")
            
            state, mask = next_state, next_mask
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_pegs.append(info["pegs_remaining"])
        
        if info.get("center_win", False):
            center_wins += 1
            wins += 1
        elif info["pegs_remaining"] == 1:
            wins += 1
    
    # Resultados
    win_rate = wins / num_episodes
    center_win_rate = center_wins / num_episodes
    avg_reward = np.mean(episode_rewards)
    avg_pegs = np.mean(episode_pegs)
    
    print(f"\nResultados:")
    print(f"  Win rate (1 ficha): {win_rate:.2%}")
    print(f"  Center win rate: {center_win_rate:.2%}")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg pegs: {avg_pegs:.2f}")
    
    return {
        "win_rate": win_rate,
        "center_win_rate": center_win_rate,
        "avg_reward": avg_reward,
        "avg_pegs": avg_pegs,
    }


def main():
    """Función principal de ejemplo."""
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    # Paso 1: Entrenar agente
    agent = train_quick_demo(episodes=1000, verbose=True)
    
    # Paso 2: Evaluar agente
    env = FrenchSolitaireEnv()
    results = evaluate_agent(agent, env, num_episodes=100)
    
    # Paso 3: Guardar checkpoint
    checkpoint_path = "checkpoints/quick_demo.pt"
    os.makedirs("checkpoints", exist_ok=True)
    agent.save(checkpoint_path)
    print(f"\n💾 Checkpoint guardado: {checkpoint_path}")
    
    # Paso 4: Ejemplo de carga
    print(f"\n📂 Ejemplo de carga del checkpoint:")
    agent_loaded = DQNAgent()
    agent_loaded.load(checkpoint_path, load_optimizer=False)
    print(f"  ✓ Modelo cargado desde {checkpoint_path}")
    print(f"  ✓ Epsilon: {agent_loaded.epsilon:.3f}")
    
    print("\n✨ Ejemplo completado!")
    print("\nPróximos pasos:")
    print("  - Entrenar más episodios: python train.py --episodes 10000")
    print("  - Evaluar con renderizado: python eval.py --checkpoint checkpoints/quick_demo.pt --render")
    print("  - Explorar tutorial interactivo: marimo edit notebooks/tutorial.py")


if __name__ == "__main__":
    main()
