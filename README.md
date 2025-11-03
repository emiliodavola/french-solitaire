# French Solitaire - Reinforcement Learning

Proyecto educativo para entrenar un agente de aprendizaje por refuerzo que resuelva el juego French Solitaire (variante europea 7×7) usando PyTorch.

## Stack

- **Python**: 3.12
- **Framework RL**: PyTorch
- **Entorno**: Gymnasium
- **Tracking**: MLflow
- **Gestión de dependencias**: Miniconda

## Setup rápido

### 1. Crear el entorno (primera vez)

```powershell
# Crear entorno desde archivo
conda env create -f environment.yml

# O crear manualmente
conda create -n french-solitaire python=3.12 -y
conda activate french-solitaire
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install gymnasium numpy mlflow matplotlib tensorboard pytest
```

### 2. Activar el entorno

```powershell
conda activate french-solitaire
```

**⚠️ IMPORTANTE**: Siempre activa el entorno antes de ejecutar cualquier comando Python.

### 3. Verificar instalación de GPU

```powershell
conda activate french-solitaire
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Comandos de desarrollo

```powershell
# Entrenamiento (cuando esté implementado)
conda activate french-solitaire
python train.py --algo dqn --episodes 10000 --save-freq 1000

# Evaluación
conda activate french-solitaire
python eval.py --model checkpoints/dqn_best.pt --episodes 100 --render

# Tests
conda activate french-solitaire
pytest tests/ -v

# MLflow UI
conda activate french-solitaire
mlflow ui --backend-store-uri file:./mlruns
```

## Estructura del proyecto (planeada)

```plaintext
.
├── envs/                     # Entornos de juego (Gymnasium)
│   └── french_solitaire_env.py
├── agent/                    # Algoritmos RL (DQN, PPO, A2C)
│   ├── dqn.py
│   ├── replay_buffer.py
│   └── networks.py
├── scripts/                  # Scripts de entrenamiento
│   └── train_dqn.py
├── tests/                    # Tests unitarios
│   └── test_env.py
├── notebooks/                # Análisis exploratorio
├── checkpoints/              # Modelos guardados
├── mlruns/                   # Experimentos MLflow
├── train.py                  # Script principal de entrenamiento
├── eval.py                   # Script de evaluación
├── environment.yml           # Dependencias conda
└── README.md
```

## Reglas del juego (French Solitaire 7×7)

- Tablero europeo de 7×7 con forma de cruz
- Fichas iniciales: 32 (todas excepto la posición central)
- Movimiento: saltar una ficha adyacente sobre un espacio vacío (la ficha saltada se elimina)
- Objetivo: dejar una sola ficha en el tablero en el centro

## Recursos

- [Gymnasium Docs](https://gymnasium.farama.org/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)

## Licencia

MIT
