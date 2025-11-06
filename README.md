# French Solitaire

Entrena un agente de **Deep Q-Learning (DQN)** para resolver el juego **French Solitaire** (7×7) usando PyTorch.

**Objetivo**: Reducir 32 fichas a 1 ficha en el centro del tablero.

## 🚀 Quick Start

### 1. Activar entorno
```powershell
conda activate french-solitaire
```

### 2. Verificar instalación
```powershell
# Verificar GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Ejecutar tests (26 tests)
python -m pytest tests/ -v
```

### 3. Entrenar modelo
```powershell
# Demo rápido (1000 episodios, ~5 min)
python examples/quick_start.py

# Entrenamiento completo (10k episodios)
python train.py --episodes 10000 --run-name my-experiment

# Ver opciones
python train.py --help
```

**Checkpoints generados:**
- `my-experiment_best.pt` → Mejor modelo (para subir a HF Hub) ⭐
- `my-experiment_final.pt` → Modelo al finalizar entrenamiento
- `my-experiment_ep001000.pt` → Checkpoints intermedios cada 1000 episodios

### 4. Evaluar modelo
```powershell
# Evaluar mejor modelo
python eval.py --checkpoint checkpoints/my-experiment_best.pt --episodes 100

# Evaluar con renderizado
python eval.py --checkpoint checkpoints/my-experiment_best.pt --episodes 10 --render
```

### 5. Visualizar experimentos (MLflow)
```powershell
mlflow ui --backend-store-uri file:./mlruns
# Abrir http://localhost:5000
```

### 6. Tutorial interactivo (Marimo)
```powershell
marimo edit notebooks/tutorial.py
```

## 📦 Stack

- **Python**: 3.12
- **RL**: PyTorch + Gymnasium
- **Tracking**: MLflow
- **Env**: Miniconda

## 🛠️ Setup (primera vez)

```powershell
# Crear entorno conda
conda env create -f environment.yml

# Activar (SIEMPRE antes de usar el proyecto)
conda activate french-solitaire
```

**⚠️ Importante**: Ejecuta `conda activate french-solitaire` antes de cualquier comando Python.

## 📊 Subir a Hugging Face Hub

```powershell
# 1. Instalar y hacer login (solo primera vez)
pip install huggingface-hub
huggingface-cli login

# 2. Subir mejor modelo
python scripts/upload_to_hf.py \
  --checkpoint checkpoints/my-experiment_best.pt \
  --repo-id tu-usuario/french-solitaire
```

**El script automáticamente:**
- Renombra `my-experiment_best.pt` → `pytorch_model.pt` (estándar HF)
- Sube checkpoint + README + código + configuración
- Crea el repo en HuggingFace si no existe



## Estructura del proyecto

```plaintext
.
├── envs/                     # Entornos de juego (Gymnasium) ✅
│   ├── __init__.py
│   └── french_solitaire_env.py
├── agent/                    # Algoritmos RL (DQN) ✅
│   ├── __init__.py
│   ├── dqn.py                # Clase DQNAgent
│   ├── networks.py           # QNetwork, DuelingQNetwork
│   └── replay_buffer.py      # ReplayBuffer, PrioritizedReplayBuffer
├── scripts/                  # Scripts de entrenamiento ✅
│   ├── train_dqn.py          # Script CLI de entrenamiento
│   └── upload_to_hf.py       # Subir modelo a Hugging Face Hub
├── tests/                    # Tests unitarios ✅
│   ├── test_env.py           # Tests del entorno
│   └── test_agent.py         # Tests del agente DQN
├── notebooks/                # Análisis exploratorio
│   └── tutorial.py           # Tutorial interactivo Marimo
├── checkpoints/              # Modelos guardados (.pt)
├── mlruns/                   # Experimentos MLflow
├── train.py                  # Entrypoint principal de entrenamiento ✅
├── eval.py                   # Script de evaluación ✅
├── environment.yml           # Dependencias conda
├── model_config.json         # Configuración del modelo (para HF Hub)
├── README_HF.md              # README para Hugging Face Hub
└── README.md
```

## 🎮 Reglas del juego

```
Tablero inicial (7×7):
      O O O
      O O O
  O O O O O O O
  O O O . O O O  ← Centro vacío
  O O O O O O O
      O O O
      O O O

Objetivo: ¡Dejar solo UNA ficha en el centro!
```

- **Movimiento**: Saltar una ficha adyacente sobre un espacio vacío (horizontal/vertical)
- **Fichas iniciales**: 32
- **Victoria**: 1 ficha en el centro (3,3)

## 🧪 Tests

```powershell
# Todos los tests (26 tests)
python -m pytest tests/ -v

# Solo entorno
python -m pytest tests/test_env.py -v

# Solo agente
python -m pytest tests/test_agent.py -v
```

## 🤝 Contribuir (Git Flow)

```powershell
# Crear feature branch desde dev
git checkout dev
git pull origin dev
git checkout -b feature/mi-mejora

# Commits semánticos
git commit -m "feat: add new feature"
git commit -m "fix: correct bug"
git commit -m "test: add tests"

# Push y PR
git push origin feature/mi-mejora
# → Abrir PR en GitHub: feature/mi-mejora → dev
```

## 📚 Recursos

- [Gymnasium](https://gymnasium.farama.org/) - API de entornos RL
- [PyTorch](https://pytorch.org/docs/stable/index.html) - Deep learning
- [MLflow](https://mlflow.org/docs/latest/index.html) - Experiment tracking

## 📄 Licencia

MIT
