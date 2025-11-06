"""
Entrypoint principal para entrenamiento de agentes de RL en French Solitaire

Uso:
    conda activate french-solitaire
    python train.py --algo dqn --episodes 10000 --lr 5e-4
"""
import sys
import os

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar y ejecutar el script de entrenamiento
from scripts.train_dqn import parse_args, train_dqn


if __name__ == "__main__":
    args = parse_args()
    train_dqn(args)
