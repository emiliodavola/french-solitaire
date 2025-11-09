---
language: en
license: mit
tags:
  - reinforcement-learning
  - deep-q-learning
  - dqn
  - french-solitaire
  - peg-solitaire
  - pytorch
  - gymnasium
  - single-solution
  - deterministic
library_name: pytorch
---

# DQN Agent for French Solitaire (7×7) — Single-Solution Deterministic Policy

## Model Description

This is a **Deep Q-Network (DQN)** agent trained to solve the **French Solitaire** puzzle (Peg Solitaire, 7×7 European variant) via a **single deterministic trajectory** learned during training. The published checkpoint represents a policy that, when evaluated en modo greedy (ε = 0), sigue una ruta canónica hasta la victoria (32 → 1 ficha en el centro). It does **not** attempt to enumerate or diversify multiple winning solutions.

### Game Rules

- **Board**: 7×7 grid with 32 valid positions (European cross shape)
- **Initial state**: All positions filled except the center (3,3)
- **Objective**: Jump pegs over adjacent pegs to remove them, leaving only 1 peg in the center
- **Valid move**: Jump horizontally or vertically over an adjacent peg into an empty space

```text
Initial board:
      O O O
      O O O
  O O O O O O O
  O O O . O O O  ← Center empty
  O O O O O O O
      O O O
      O O O
```text
Goal:
      . . .
      . . .
  . . . . . . .
  . . . O . . .  ← One peg in center
  . . . . . . .
      . . .
      . . .
```

## Model Architecture

- **Algorithm**: Double DQN with Experience Replay and Target Network
- **Network**: 3-layer fully connected neural network
  - Input: 49-dimensional state (7×7 board flattened)
  - Hidden layers: 128 neurons each with ReLU activation
  - Output: 100 Q-values (action space)
- **Framework**: PyTorch 2.x
- **Action masking**: Only valid moves are considered during action selection

### Hyperparameters

- Learning rate: 5e-4
- Gamma (discount factor): 0.99
- Epsilon decay: 0.995 (start: 1.0, end: 0.01)
- Batch size: 64
- Replay buffer size: 10,000
- Target network update frequency: 100 steps

### Reward Function

- **+100**: Victory (1 peg in center)
- **+50**: 1 peg remaining (but not in center)
- **+1**: Progress (peg removed)
- **-10**: Invalid move
- **-50**: No valid moves left (defeat)

## Training Details

- **Episodes**: 10,000
- **Training time**: ~20 minutes on NVIDIA GPU (CUDA 12.1)
- **Win rate**: 100.0% (1 peg remaining)
- **Center win rate**: 100.0% (perfect victories)
- **Average pegs remaining**: 1.0

Training was logged with **MLflow** and tracked in `./mlruns`.

## Determinism / Single-Solution Note

During evaluation we force `epsilon = 0.0`, yielding a greedy policy. Given fixed weights and initial board, action selection is deterministic (tie-breaking handled by `argmax`). If you need multiple trajectories or stochastic solution sampling, you should train or evaluate with a modified script (e.g. softmax over Q-values or ε>0) — not included in this single-solution release.

To keep the repository focused, tags include `single-solution` and `deterministic`.

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/emiliodavola/french-solitaire.git
cd french-solitaire

# Create conda environment
conda env create -f environment.yml
conda activate french-solitaire
```

### Evaluate the model

```bash
# Download checkpoint from this repo (single-solution deterministic)
# The canonical model file is: pytorch_model.pt

# Run evaluation (100 episodes)
conda activate french-solitaire
python eval.py --checkpoint pytorch_model.pt --episodes 100

# With visual rendering (shows all steps)
python eval.py --checkpoint pytorch_model.pt --episodes 10 --render
```

### Load in Python

```python
import torch
from agent.dqn import DQNAgent
from envs.french_solitaire_env import FrenchSolitaireEnv

# Create environment and agent
env = FrenchSolitaireEnv()
agent = DQNAgent(state_dim=49, action_dim=100)

# Load checkpoint (pytorch_model.pt from this repo)
agent.load("pytorch_model.pt", load_optimizer=False)
agent.epsilon = 0.0  # Greedy (no exploration)

# Play one episode
state, info = env.reset()
done = False

while not done:
    mask = info.get("action_mask")
    action = agent.select_action(state, action_mask=mask, training=False)
    state, reward, done, truncated, info = env.step(action)
    print(env.render())

print(f"Pegs remaining: {info['pegs_remaining']}")
print(f"Victory: {info.get('center_win', False)}")
```

## Performance

| Metric | Value |
|--------|-------|
| Win rate (1 peg) | 100.0% |
| Center win rate (perfect) | 100.0% |
| Avg. reward per episode | 130.0 |
| Avg. pegs remaining | 1.0 |
| Avg. steps per episode | 31.0 |

## Limitations

This release purposely reflects a **single deterministic solution path**:

- Trained specifically for the 7×7 French Solitaire variant.
- Does not expose multiple diverse winning trajectories.
- Action space is fixed at 100 pre-computed geometric moves.
- Performance metrics reported here correspond to greedy (ε=0) evaluation only.

## Future Improvements / Multi-Solution Roadmap

Potential extensions (not part of this checkpoint):

- [ ] Stochastic evaluation (softmax over Q or ε-sampling) to collect multiple solution paths.
- [ ] Exact DFS solver for enumerating all distinct winning trajectories (with symmetry pruning).
- [ ] Dueling DQN architecture.
- [ ] Prioritized Experience Replay.
- [ ] Curriculum learning.
- [ ] Board symmetry data augmentation.
- [ ] Extended training (50k–100k episodes) and hyperparameter tuning (Ray Tune).

## Citation

If you use this model or code, please cite:

```bibtex
@misc{french-solitaire-dqn-single-solution,
  author = {Emilio Davola},
  title = {DQN Agent for French Solitaire - Single-Solution Deterministic Policy},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/emiliodavola/french-solitaire-dqn-single-solution}}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2015)](https://arxiv.org/abs/1509.06461)

## Repository

Full code and training scripts: [https://github.com/emiliodavola/french-solitaire](https://github.com/emiliodavola/french-solitaire)
