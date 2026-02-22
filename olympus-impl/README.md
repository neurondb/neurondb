# Olympus Implementation

PyTorch implementation of **Olympus: Multi-Resolution Neural Memory with Selective Consolidation for Sequence Modeling**, extending Titans-style test-time learnable memory with:

- **Multi-Resolution Hierarchy (MRH)**: Fast / Mid / Slow memory scales
- **Importance-Weighted Selective Forgetting (ISF)**: Diagonal Fisher-based per-parameter decay
- **Contrastive associative loss**: InfoNCE-style term
- **Curvature-aware inner-loop**: RMSProp-style per-parameter step sizes
- **Consolidation**: Fast → Mid → Slow transfer

## Setup

```bash
cd olympus-impl
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

## Usage

- **Sanity check (memory updates)**:
  ```bash
  python -m pytest tests/ -v
  ```

- **Language modeling (WikiText-103)**:
  ```bash
  python experiments/lm/train_lm.py --config experiments/lm/config_small.yaml
  ```

- **Needle-in-a-haystack evaluation**:
  ```bash
  python experiments/niah/run_niah.py --model_path <path> --context_lengths 2048 4096 8192 16384
  ```

- **Plotting (after runs)**:
  ```bash
  python scripts/plot_results.py --lm_logdir runs/lm --niah_logdir runs/niah --outdir plots
  ```

## Project layout

- `src/olympus/` – Core Olympus memory and MAC/MAG/MAL variants
- `experiments/lm/` – Language modeling data, training, configs
- `experiments/niah/` – S-NIAH task generation and evaluation
- `tests/` – Unit tests for memory updates
- `scripts/` – Plotting and reporting
