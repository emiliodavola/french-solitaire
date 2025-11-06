"""
Script de entrenamiento DQN para French Solitaire

Uso:
    conda activate french-solitaire
    python scripts/train_dqn.py --episodes 10000 --lr 5e-4 --gamma 0.99
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

# Añadir el directorio raíz al path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.french_solitaire_env import FrenchSolitaireEnv
from agent.dqn import DQNAgent


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena un agente DQN en French Solitaire"
    )
    
    # Parámetros de entrenamiento
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Número de episodios de entrenamiento (default: 10000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Máximo de pasos por episodio (default: 200)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100,
        help="Frecuencia de evaluación en episodios (default: 100)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1000,
        help="Frecuencia de guardado de checkpoints (default: 1000)",
    )
    
    # Hiperparámetros del agente
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Factor de descuento (default: 0.99)"
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Epsilon inicial (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.01,
        help="Epsilon final (default: 0.01)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Tasa de decay de epsilon (default: 0.995)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Tamaño del batch (default: 64)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Tamaño del replay buffer (default: 10000)",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=100,
        help="Frecuencia de actualización de target network (default: 100)",
    )
    
    # Dispositivo y logging
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo: 'cuda' o 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directorio para guardar checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre de la corrida (default: dqn-TIMESTAMP)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Desactivar logging a MLflow",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla aleatoria (default: None)"
    )
    
    return parser.parse_args()


def train_dqn(args):
    """
    Entrena el agente DQN.
    
    Args:
        args: argumentos parseados de argparse
    
    Returns:
        dict: resultados del entrenamiento
    """
    # Semilla aleatoria
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Crear directorio de checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Nombre de la corrida
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"dqn-{timestamp}"
    
    # MLflow (opcional)
    mlflow_enabled = False
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("french-solitaire")
            mlflow_enabled = True
        except ImportError:
            print("[Warning] MLflow no disponible, entrenamiento sin logging")
    
    # Crear entorno y agente
    env = FrenchSolitaireEnv()
    agent = DQNAgent(
        state_dim=49,
        action_dim=100,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=args.device,
    )
    
    print(f"=== Entrenamiento DQN: {args.run_name} ===")
    print(f"Dispositivo: {agent.device}")
    print(f"Episodios: {args.episodes}")
    print(f"Hiperparámetros: lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size}")
    print()
    
    # Métricas
    episode_rewards = []
    episode_losses = []
    episode_pegs = []
    wins = 0
    best_avg_reward = float('-inf')
    best_episode = 0
    
    # MLflow logging
    if mlflow_enabled:
        with mlflow.start_run(run_name=args.run_name) as run:
            # Log de parámetros
            mlflow.log_params({
                "algorithm": "DQN",
                "learning_rate": args.lr,
                "gamma": args.gamma,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "epsilon_decay": args.epsilon_decay,
                "batch_size": args.batch_size,
                "buffer_size": args.buffer_size,
                "target_update_freq": args.target_update_freq,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
            })
            
            # Loop de entrenamiento
            for episode in range(args.episodes):
                state, info = env.reset()
                mask = info.get("action_mask")
                total_reward = 0
                done = False
                losses = []
                steps = 0
                
                while not done and steps < args.max_steps:
                    # Seleccionar acción
                    action = agent.select_action(state, action_mask=mask, training=True)
                    
                    # Ejecutar acción
                    next_state, reward, done, truncated, info = env.step(action)
                    next_mask = info.get("action_mask")
                    
                    # Guardar en replay buffer
                    agent.replay_buffer.push(
                        state, action, reward, next_state, done, next_mask
                    )
                    
                    # Entrenar
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                    
                    # Avanzar
                    state, mask = next_state, next_mask
                    total_reward += reward
                    steps += 1
                
                # Decay epsilon
                agent.decay_epsilon()
                
                # Métricas del episodio
                episode_rewards.append(total_reward)
                episode_losses.append(np.mean(losses) if losses else 0)
                episode_pegs.append(info["pegs_remaining"])
                if info.get("center_win", False):
                    wins += 1
                
                # Log a MLflow cada episodio
                mlflow.log_metrics({
                    "reward": total_reward,
                    "loss": np.mean(losses) if losses else 0,
                    "epsilon": agent.epsilon,
                    "pegs": info["pegs_remaining"],
                    "steps": steps,
                }, step=episode)
                
                # Log adicional cada eval_freq episodios
                if (episode + 1) % args.eval_freq == 0:
                    avg_reward = np.mean(episode_rewards[-args.eval_freq:])
                    avg_pegs = np.mean(episode_pegs[-args.eval_freq:])
                    win_rate = wins / (episode + 1)
                    
                    mlflow.log_metrics({
                        "avg_reward": avg_reward,
                        "avg_pegs": avg_pegs,
                        "win_rate": win_rate,
                    }, step=episode)
                    
                    print(f"Ep {episode + 1}/{args.episodes} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Pegs: {avg_pegs:.2f} | "
                          f"Win Rate: {win_rate:.2%} | "
                          f"Epsilon: {agent.epsilon:.3f}")
                    
                    # Guardar best checkpoint si mejora
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_episode = episode + 1
                        best_checkpoint = os.path.join(
                            args.checkpoint_dir,
                            f"{args.run_name}_best.pt"
                        )
                        agent.save(best_checkpoint)
                        mlflow.log_artifact(best_checkpoint)
                        print(f"  → Nuevo mejor modelo guardado: {best_checkpoint} (reward: {avg_reward:.2f})")
                
                # Guardar checkpoint intermedio
                if (episode + 1) % args.save_freq == 0:
                    checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f"{args.run_name}_ep{episode + 1:06d}.pt"
                    )
                    agent.save(checkpoint_path)
                    mlflow.log_artifact(checkpoint_path)
                    print(f"  → Checkpoint guardado: {checkpoint_path}")
            
            # Guardar checkpoint final
            final_checkpoint = os.path.join(
                args.checkpoint_dir,
                f"{args.run_name}_final.pt"
            )
            agent.save(final_checkpoint)
            mlflow.log_artifact(final_checkpoint)
            
            print("\n=== Entrenamiento completado ===")
            print(f"Win rate final: {wins / args.episodes:.2%}")
            print(f"Mejor modelo: {args.run_name}_best.pt (episodio {best_episode}, reward: {best_avg_reward:.2f})")
            print(f"Checkpoint final: {final_checkpoint}")
            print(f"MLflow run ID: {run.info.run_id}")
    
    else:
        # Entrenamiento sin MLflow (mismo loop)
        for episode in range(args.episodes):
            state, info = env.reset()
            mask = info.get("action_mask")
            total_reward = 0
            done = False
            losses = []
            steps = 0
            
            while not done and steps < args.max_steps:
                action = agent.select_action(state, action_mask=mask, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                next_mask = info.get("action_mask")
                
                agent.replay_buffer.push(
                    state, action, reward, next_state, done, next_mask
                )
                
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                
                state, mask = next_state, next_mask
                total_reward += reward
                steps += 1
            
            agent.decay_epsilon()
            
            episode_rewards.append(total_reward)
            episode_losses.append(np.mean(losses) if losses else 0)
            episode_pegs.append(info["pegs_remaining"])
            if info.get("center_win", False):
                wins += 1
            
            if (episode + 1) % args.eval_freq == 0:
                avg_reward = np.mean(episode_rewards[-args.eval_freq:])
                avg_pegs = np.mean(episode_pegs[-args.eval_freq:])
                win_rate = wins / (episode + 1)
                
                print(f"Ep {episode + 1}/{args.episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Pegs: {avg_pegs:.2f} | "
                      f"Win Rate: {win_rate:.2%} | "
                      f"Epsilon: {agent.epsilon:.3f}")
                
                # Guardar best checkpoint si mejora
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_episode = episode + 1
                    best_checkpoint = os.path.join(
                        args.checkpoint_dir,
                        f"{args.run_name}_best.pt"
                    )
                    agent.save(best_checkpoint)
                    print(f"  → Nuevo mejor modelo guardado: {best_checkpoint} (reward: {avg_reward:.2f})")
            
            if (episode + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f"{args.run_name}_ep{episode + 1:06d}.pt"
                )
                agent.save(checkpoint_path)
                print(f"  → Checkpoint guardado: {checkpoint_path}")
        
        final_checkpoint = os.path.join(
            args.checkpoint_dir,
            f"{args.run_name}_final.pt"
        )
        agent.save(final_checkpoint)
        
        print("\n=== Entrenamiento completado ===")
        print(f"Win rate final: {wins / args.episodes:.2%}")
        print(f"Mejor modelo: {args.run_name}_best.pt (episodio {best_episode}, reward: {best_avg_reward:.2f})")
        print(f"Checkpoint final: {final_checkpoint}")
    
    return {
        "rewards": episode_rewards,
        "losses": episode_losses,
        "pegs": episode_pegs,
        "wins": wins,
        "win_rate": wins / args.episodes,
    }


if __name__ == "__main__":
    args = parse_args()
    train_dqn(args)
