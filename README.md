# French Solitaire

Entrena un agente de **Deep Q-Learning (DQN)** para el juego **French Solitaire** (7×7) usando PyTorch.

**Objetivo**: Reducir 32 fichas a 1 ficha en el centro del tablero.

## 🎓 Alcance y propósito

Este repositorio es un **ejercicio pedagógico de Reinforcement Learning**: implementa desde cero un entorno compatible con Gymnasium y un agente DQN para estudiar cómo se construyen y se entrenan.

**No es un solver de French Solitaire.** El problema es determinista y de estado finito, así que un solver de búsqueda exacta (DFS/BFS con memoización) lo resuelve de forma óptima en milisegundos. Acá el objetivo es aprender RL, no resolver el puzzle por la vía más eficiente. Que el agente encuentre una trayectoria ganadora sobre un tablero inicial fijo es un resultado pedagógico, no una demostración de que RL sea el método adecuado para este problema.

**Limitaciones conocidas y trabajo pendiente**: ver los [issues abiertos](https://github.com/emiliodavola/french-solitaire/issues).

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
# Evaluar mejor modelo (solo métricas)
python eval.py --checkpoint checkpoints/my-experiment_best.pt --episodes 100

# Evaluar con renderizado visual (muestra tablero en cada paso)
python eval.py --checkpoint checkpoints/my-experiment_best.pt --episodes 10 --render

# Renderizado + información adicional (estado interno, movimientos válidos)
python eval.py --checkpoint checkpoints/my-experiment_best.pt --episodes 10 --render --verbose

# Ver todas las opciones
python eval.py --help
```

**Nota:** `--render` ahora muestra la secuencia completa de movimientos paso a paso, no solo el tablero inicial y final.

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

# 2. Subir mejor modelo (detecta automáticamente tu token)
python scripts/upload_to_hf.py \
  --checkpoint checkpoints/my-experiment_best.pt \
  --repo-id tu-usuario/french-solitaire
```

**El script automáticamente:**
- Detecta el token de `huggingface-cli login` (no necesitas pasarlo manualmente)
- Renombra `my-experiment_best.pt` → `pytorch_model.pt` (estándar HF)
- Limpia archivos `__pycache__` del repositorio
- Sube checkpoint + README + código + configuración
- Crea el repo en HuggingFace si no existe

**Opciones adicionales:**
```powershell
# Crear repositorio privado
python scripts/upload_to_hf.py \
  --checkpoint checkpoints/my-experiment_best.pt \
  --repo-id tu-usuario/french-solitaire \
  --private

# Mensaje de commit personalizado
python scripts/upload_to_hf.py \
  --checkpoint checkpoints/my-experiment_best.pt \
  --repo-id tu-usuario/french-solitaire \
  --commit-message "Update model - improved training"
```



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

- **Tablero**: 33 posiciones válidas (cruz 3-3-7-7-7-3-3); 16 celdas quedan fuera del tablero
- **Movimiento**: Saltar una ficha adyacente sobre un espacio vacío (horizontal/vertical)
- **Fichas iniciales**: 32 (33 posiciones, con el centro vacío)
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
