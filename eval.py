"""
Script de evaluación de modelos entrenados para French Solitaire

Uso:
    conda activate french-solitaire
    python eval.py --checkpoint checkpoints/dqn_final.pt --episodes 100 --render
"""
import argparse
import os
import sys

import numpy as np
import torch

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from envs.french_solitaire_env import FrenchSolitaireEnv
from agent.dqn import DQNAgent


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evalúa un agente DQN entrenado en French Solitaire"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta del checkpoint del modelo (.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Número de episodios de evaluación (default: 100)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Máximo de pasos por episodio (default: 200)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Renderizar el juego durante la evaluación",
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=10,
        help="Frecuencia de renderizado (cada N episodios, si --render activado)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo: 'cuda' o 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria (default: None)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprimir detalles de cada episodio",
    )
    
    return parser.parse_args()


def evaluate_agent(env, agent, num_episodes=100, max_steps=200, render=False, render_freq=10, verbose=False):
    """
    Evalúa el agente entrenado.
    
    Args:
        env: entorno del juego
        agent: agente DQN
        num_episodes (int): número de episodios
        max_steps (int): máximo de pasos por episodio
        render (bool): si renderizar el juego
        render_freq (int): frecuencia de renderizado
        verbose (bool): imprimir detalles
    
    Returns:
        dict: estadísticas de evaluación
    """
    agent.epsilon = 0.0  # Sin exploración (modo greedy)
    agent.q_network.eval()  # Modo evaluación
    
    wins = 0
    center_wins = 0
    episode_rewards = []
    episode_pegs = []
    episode_steps = []
    
    print(f"=== Evaluando agente (ε=0, {num_episodes} episodios) ===\n")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        mask = info.get("action_mask")
        total_reward = 0
        done = False
        steps = 0
        
        # Renderizar si aplica
        should_render = render and ((episode % render_freq == 0) or verbose)
        if should_render:
            print(f"--- Episodio {episode + 1}/{num_episodes} ---")
            print(env.render())
        
        while not done and steps < max_steps:
            # Seleccionar acción (greedy)
            action = agent.select_action(state, action_mask=mask, training=False)
            
            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)
            next_mask = info.get("action_mask")
            
            # Avanzar
            state, mask = next_state, next_mask
            total_reward += reward
            steps += 1
            
            if should_render and verbose:
                print(f"Acción: {action} | Recompensa: {reward:+.1f} | Fichas: {info['pegs_remaining']}")
                if done:
                    print(env.render())
        
        # Registrar métricas
        episode_rewards.append(total_reward)
        episode_pegs.append(info["pegs_remaining"])
        episode_steps.append(steps)
        
        if info.get("center_win", False):
            center_wins += 1
            wins += 1
        elif info["pegs_remaining"] == 1:
            wins += 1
        
        if should_render:
            outcome = "VICTORIA ✅" if info.get("center_win", False) else ("1 ficha ⚠️" if info["pegs_remaining"] == 1 else "DERROTA ❌")
            print(f"Resultado: {outcome} | Reward: {total_reward:.1f} | Pasos: {steps}\n")
    
    # Estadísticas finales
    win_rate = wins / num_episodes
    center_win_rate = center_wins / num_episodes
    avg_reward = np.mean(episode_rewards)
    avg_pegs = np.mean(episode_pegs)
    avg_steps = np.mean(episode_steps)
    
    print("=== Resultados de evaluación ===")
    print(f"Episodios: {num_episodes}")
    print(f"Win rate (1 ficha): {win_rate:.2%}")
    print(f"Center win rate (ficha en centro): {center_win_rate:.2%}")
    print(f"Recompensa promedio: {avg_reward:.2f}")
    print(f"Fichas promedio restantes: {avg_pegs:.2f}")
    print(f"Pasos promedio: {avg_steps:.1f}")
    
    return {
        "win_rate": win_rate,
        "center_win_rate": center_win_rate,
        "avg_reward": avg_reward,
        "avg_pegs": avg_pegs,
        "avg_steps": avg_steps,
        "rewards": episode_rewards,
        "pegs": episode_pegs,
        "steps": episode_steps,
    }


def main():
    """Función principal de evaluación."""
    args = parse_args()
    
    # Semilla aleatoria
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Verificar que el checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint no encontrado: {args.checkpoint}")
        sys.exit(1)
    
    # Crear entorno
    env = FrenchSolitaireEnv(render_mode="human" if args.render else None)
    
    # Crear agente
    agent = DQNAgent(
        state_dim=49,
        action_dim=100,
        device=args.device,
    )
    
    # Cargar checkpoint
    print(f"Cargando checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint, load_optimizer=False)
    print(f"Dispositivo: {agent.device}\n")
    
    # Evaluar
    results = evaluate_agent(
        env,
        agent,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        render_freq=args.render_freq,
        verbose=args.verbose,
    )
    
    return results


if __name__ == "__main__":
    main()
