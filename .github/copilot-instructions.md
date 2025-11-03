## Propósito

Este repositorio entrena un agente de **aprendizaje por refuerzo (RL)** para resolver el juego **French Solitaire** (variante europea 7×7, también conocido como Peg Solitaire o "Senku"). El proyecto es educativo: implementa el entorno del juego compatible con Gym/Gymnasium y algoritmos de RL usando **PyTorch** (DQN, PPO, A2C u otros) para aprender sobre RL mientras se entrena un agente que realice movimientos óptimos.

**Stack principal**: Python 3.12 + PyTorch (GPU con conda) + Gymnasium + MLflow

**Entorno**: Miniconda con entorno `french-solitaire` (Python 3.12)

Última actualización: 2025-11-03

## ⚠️ REGLA CRÍTICA PARA AGENTES IA ⚠️

**SIEMPRE activa el entorno conda antes de ejecutar CUALQUIER comando Python/pip/pytest/mlflow.**

Usar en PowerShell:
```powershell
conda activate french-solitaire
# ... luego tu comando
```

O ejecutar directamente con `conda run`:
```powershell
conda run -n french-solitaire python script.py
conda run -n french-solitaire pip install paquete
conda run -n french-solitaire pytest tests/
```

**Nunca ejecutes comandos Python sin activar el entorno primero.** Esto evita errores de importación, versiones incorrectas y gasto innecesario de tokens.

## Arquitectura esperada (guía para agentes IA)

Cuando explores el código, busca y respeta esta estructura típica de proyectos RL:

1. **Entorno del juego** (`env/` o `envs/french_solitaire_env.py`)
   - Clase que hereda de `gym.Env` o `gymnasium.Env`
   - Métodos clave: `reset()`, `step(action)`, `render()`, `_get_obs()`, `_is_valid_move()`
   - Espacios de observación y acción (`observation_space`, `action_space`)
   - Lógica de recompensas: penalización por movimiento inválido, recompensa por reducir fichas, bonus por victoria

2. **Agente/Algoritmo RL** (`agent/` o `models/`)
   - Implementaciones de algoritmos usando **PyTorch** (DQN, PPO, A2C) - preferir implementaciones custom para aprendizaje sobre frameworks como Stable-Baselines3
   - Redes neuronales (policy network, value network) con `torch.nn.Module`
   - Replay buffers (para DQN), optimizadores (`torch.optim.Adam`), funciones de pérdida (`F.mse_loss`, `F.smooth_l1_loss`)

3. **Scripts de entrenamiento** (`train.py`, `scripts/train_*.py`)
   - Bucles de entrenamiento con logging de métricas a **MLflow** (recompensa acumulada, tasa de victoria, epsilon-decay, pérdida)
   - Checkpointing de modelos cada N episodios (guardar state_dict de PyTorch)
   - Configuración de hiperparámetros (learning rate, gamma, batch size, exploration strategy)

4. **Evaluación y visualización** (`eval.py`, `play.py`, `notebooks/`)
   - Scripts para cargar modelos entrenados (checkpoints `.pt` de PyTorch) y evaluar performance
   - Renderizado del tablero (CLI con ASCII art como prioridad, matplotlib/pygame opcional)
   - Notebooks Jupyter para análisis de curvas de aprendizaje (visualización **secundaria**, no bloqueante para desarrollo inicial)

5. **Configuración y dependencias**
   - `environment.yml` de conda con Python 3.12, PyTorch GPU (CUDA 12.1), y dependencias de RL
   - `requirements.txt` complementario para paquetes pip-only (si aplica)
   - Archivos de config (`config.yaml`, `hyperparams.json`) para experimentos reproducibles
   - **No usar Stable-Baselines3** — implementaciones custom con PyTorch para aprendizaje práctico

## Comandos de desarrollo (Python 3.12 + conda)

```powershell
# PASO 0: SIEMPRE activar el entorno primero (obligatorio)
conda activate french-solitaire

# Instalación de dependencias (PyTorch GPU + RL)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install gymnasium numpy mlflow matplotlib tensorboard pytest

# Entrenamiento (ejemplo)
conda activate french-solitaire  # Verificar que está activado
python train.py --algo dqn --episodes 10000 --save-freq 1000

# Evaluación
conda activate french-solitaire
python eval.py --model checkpoints/dqn_best.pt --episodes 100 --render

# Pruebas unitarias
conda activate french-solitaire
pytest tests/ -v

# MLflow UI (visualización de experimentos)
conda activate french-solitaire
mlflow ui --backend-store-uri file:./mlruns

# Alternativa: usar conda run (no requiere activación previa)
conda run -n french-solitaire python train.py --algo dqn --episodes 1000
conda run -n french-solitaire pytest tests/
conda run -n french-solitaire mlflow ui
```

**Importante para agentes IA**: 
- **NUNCA** ejecutes `python`, `pip`, `pytest`, `mlflow` sin prefijo `conda activate french-solitaire` o `conda run -n french-solitaire`.
- Si ves un error de importación (`ModuleNotFoundError`), verifica primero si activaste el entorno.
- Si no hay `train.py` aún, créalo usando PyTorch con implementaciones custom (evitar wrappers de Stable-Baselines3).

## Patrones y convenciones de RL a seguir

- **Entorno Gym/Gymnasium**: todas las clases de entorno deben heredar de `gym.Env` y registrar espacios `observation_space` y `action_space` correctamente
- **Recompensas consistentes**: documentar la función de recompensa en el código (ej. `-1` por movimiento inválido, `+1` por reducir fichas, `+100` por victoria)
- **Reproducibilidad**: usar semillas fijas en scripts de entrenamiento (`np.random.seed()`, `torch.manual_seed()`)
- **Checkpoints**: guardar modelos cada N episodios con timestamp y métricas (ej. `model_dqn_ep10000_reward-45.3.pt` usando `torch.save(model.state_dict(), path)`)
- **Logging**: usar **MLflow** para tracking de experimentos (métricas, parámetros, artefactos); complementar con gráficas matplotlib para análisis exploratorio
- **Tests**: priorizar tests del entorno (validación de reglas del juego, espacios de acción/observación) antes que del modelo RL

## Integración y dependencias

- Si hay archivos de CI (GitHub Actions en `.github/workflows/`), leerlos y respetar los pasos de build/test que ahí se definan.
- Evitar cambios que rompan un flujo de CI existente; si se propone un cambio que lo afecta, incluir ajustes al workflow y pruebas locales que demuestren que funciona.

## Workflow de Git (Git Flow)

Este proyecto sigue la metodología **Git Flow** para organizar branches y commits:

### Estructura de branches

- **`main`**: producción/releases estables — solo merges desde `dev` o `hotfix/*`
- **`dev`**: rama de desarrollo principal — integración de features
- **`feature/*`**: nuevas funcionalidades (ej. `feature/dqn-agent`, `feature/french-solitaire-env`)
- **`hotfix/*`**: correcciones urgentes en `main` (ej. `hotfix/fix-reward-calculation`)

### Reglas para agentes IA

1. **Nunca commitear directamente a `main` o `dev`**
2. **Crear feature branch** desde `dev`:
   ```powershell
   git checkout dev
   git pull origin dev
   git checkout -b feature/nombre-descriptivo
   ```

3. **Commits semánticos** (Conventional Commits):
   - `feat:` nueva funcionalidad (ej. `feat: implement DQN replay buffer`)
   - `fix:` corrección de bugs (ej. `fix: correct reward calculation in step()`)
   - `docs:` documentación (ej. `docs: add training instructions to README`)
   - `test:` añadir/modificar tests (ej. `test: add unit tests for french_solitaire_env`)
   - `refactor:` refactorización sin cambios funcionales
   - `chore:` cambios menores (dependencias, configs)

4. **Commits atómicos**: un commit = una responsabilidad (no mezclar "añadir DQN + fix bug + docs")

5. **PR desde feature branch a `dev`**:
   ```powershell
   git push origin feature/nombre-descriptivo
   # Abrir PR en GitHub: feature/nombre-descriptivo → dev
   ```

### Ejemplo de workflow completo

```powershell
# 1. Crear feature branch
git checkout dev
git pull origin dev
git checkout -b feature/implement-dqn-agent

# 2. Hacer cambios y commits semánticos
conda activate french-solitaire
# ... editar archivos ...
git add agent/dqn.py agent/replay_buffer.py
git commit -m "feat: implement DQN agent with experience replay"

# 3. Más commits si es necesario
git add tests/test_dqn.py
git commit -m "test: add unit tests for DQN agent"

# 4. Push y abrir PR
git push origin feature/implement-dqn-agent
# Abrir PR en GitHub hacia dev
```

### Mensajes de commit (ejemplos)

- ✅ `feat: add French Solitaire 7x7 environment with Gymnasium API`
- ✅ `fix: resolve invalid move detection in _is_valid_move()`
- ✅ `test: add integration tests for training loop`
- ✅ `docs: document reward function in env docstring`
- ❌ `update code` (demasiado vago)
- ❌ `fix bug` (no describe qué bug)
- ❌ `wip` (work in progress no debería ser permanente)

## Ejemplos concretos para este proyecto

- Si creas `envs/french_solitaire_env.py`: implementar el tablero como matriz 7×7 con posiciones válidas marcadas; método `_is_valid_move()` debe validar saltos según reglas del juego
- Si implementas DQN: usar replay buffer de al menos 10k transiciones; red neuronal puede ser simple (2-3 capas fully-connected) dado el espacio de estados discreto
- Renderizado: priorizar ASCII art simple en CLI antes de implementar visualización gráfica (más fácil para debugging rápido)
- MLflow tracking: usar `mlflow.log_param()` para hiperparámetros, `mlflow.log_metric()` cada N episodios, y `mlflow.pytorch.log_model()` para checkpoints

## Qué incluir en un PR propuesto por un agente

1. Descripción corta del cambio y por qué (3-5 líneas).
2. Lista de archivos editados y el objetivo por cada uno.
3. Comandos ejecutados localmente (entrenamiento de prueba, tests unitarios) y salida resumida (ej. "Entrenado 1000 episodios, recompensa media: -15.3").
4. Si añades un algoritmo nuevo: incluir hiperparámetros usados y justificación breve.
5. Si modificas el entorno: ejecutar tests del entorno para validar que `observation_space`, `action_space` y lógica de recompensas funcionen correctamente.
6. **Verificar Git Flow**: PR debe ir desde `feature/*` hacia `dev` (nunca directamente a `main`).
7. **Commits semánticos**: todos los commits deben seguir el formato `tipo: descripción` (ver sección Workflow de Git).

## Señales para pedir ayuda humana

- Decisiones de arquitectura del entorno (ej. ¿representar el tablero como matriz o grafo?)
- Elección entre algoritmos RL (¿DQN vs PPO vs A2C para este problema?)
- Debugging de convergencia lenta o divergencia durante entrenamiento
- Cambios que afecten la definición de la función de recompensa (alto impacto en el aprendizaje)

## Cierre y feedback

Esta guía se enfoca en proyectos de **aprendizaje por refuerzo aplicado a French Solitaire** usando **PyTorch** para implementaciones educativas. Cuando subas código (`train.py`, `envs/`, etc.), lo actualizaré con referencias exactas a tus clases, funciones y comandos reales.
