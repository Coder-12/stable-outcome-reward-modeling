---
layout: default
title: Stable Outcome Reward Modeling via Pairwise Preference Learning
---

<div align="center">

# Stable Outcome Reward Modeling via Pairwise Preference Learning

**A robust, reproducible framework for training Outcome Reward Models (ORMs) for agentic reasoning systems**

[![Paper](https://img.shields.io/badge/Paper-ArXiv_(submitted,_under_moderation)-orange)](https://arxiv.org)
[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/LossFunctionLover/pairwise-orm-model)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HuggingFace-blue)](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This repository provides the complete implementation for training and evaluating **Pairwise Outcome Reward Models (ORMs)** as described in our paper. The pairwise formulation prioritizes stability and reproducibility under minimal supervision, achieving **96.3% pairwise accuracy** with training convergence in just **800 steps (~10 minutes on a single GPU)**.

## AI Safety Context

This work is motivated by challenges in scalable oversight for increasingly
capable language models. In agentic training loops (e.g., best-of-N sampling,
tree search, iterative refinement), reward models act as control signals rather
than static evaluators.

This project focuses on identifying and enforcing outcome reward model
properties—stability, calibration, and robustness—that are necessary for safe
deployment in such settings.

This makes outcome reward modeling not just a performance component, but a safety-critical subsystem.

### Key Results

| Metric | Value |
|--------|-------|
| **Pairwise Accuracy** | 96.3% |
| Bootstrap 90% CI | [95.3%, 97.1%] |
| Training Steps | 800 |
| Anti-symmetry Correlation | -0.998 |
| Length Robustness | 95.5% – 99.7% |

---

## Why Pairwise ORMs?

Traditional pointwise ORMs suffer from:
- High variance across random seeds
- Sensitivity to hyperparameters
- Poor calibration on out-of-distribution inputs

Our pairwise formulation addresses these challenges by learning **relative preferences** rather than absolute scores, enabling:

- **Label efficiency** — Comparative judgments are easier to obtain than absolute scores
- **Robustness** — Less sensitive to annotator disagreement and scale ambiguity
- **Stability** — Avoids collapse and calibration issues common in pointwise regression
- **Compatibility** — Integrates naturally with agentic frameworks (best-of-N sampling, tree search)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Coder-12/stable-outcome-reward-modeling.git
cd stable-outcome-reward-modeling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- See `requirements.txt` for full dependencies

---

## Quick Start

### Training

```bash
python src/training/train_pairwise_orm.py --config configs/pairwise_orm.yaml
```

### Evaluation

```bash
python src/eval/eval_pairwise_orm.py \
  --checkpoint runs/pairwise_orm/orm_pairwise_final.pt \
  --base_model facebook/opt-1.3b \
  --test_path data/processed/orm_pairwise_test.jsonl
```

---

## Architecture

```
Input Text (Reasoning Trace)
         ↓
[Frozen Base LM Encoder]  ← Pre-trained, frozen during training
         ↓
    [Mean Pooling]
         ↓
 [Lightweight MLP Head]   ← Only these parameters are trained
         ↓
  Scalar Reward Score
```

**Design Philosophy:**
- **Frozen encoder** — Leverages pre-trained representations, reduces overfitting
- **Lightweight head** — <1M trainable parameters for stability
- **Minimal architecture** — Prioritizes reproducibility over complexity

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 1e-4 with cosine decay |
| Batch Size | 32 pairs |
| Gradient Clipping | Max norm 1.0 |
| Training Steps | 800 |
| Warmup | 50 steps |
| Precision | FP16 |

**Loss Function:**
```python
L = -log(σ(f(x_chosen) - f(x_rejected)))
```

---

## Dataset

The model is trained on the **ORM Pairwise Preference Pairs** dataset — a carefully curated collection of reasoning trace preferences.

| Split | Pairs | Negatives/Positive |
|-------|-------|-------------------|
| Train | 41,656 | 8 |
| Validation | 1,144 | 4 |
| Test | 1,232 | 4 |

**Data Format:**
```json
{
  "chosen": "Step-by-step reasoning trace (correct)",
  "rejected": "Step-by-step reasoning trace (incorrect)",
  "meta": {
    "chosen_chain_id": "pos-xxxxx",
    "rejected_chain_id": "synv1-xxxxx",
    "chosen_label": 1,
    "rejected_label": 0
  }
}
```

**Quality Metrics (Source Pointwise Dataset):**
- Pearson correlation: r = 0.87
- Spearman correlation: ρ = 0.83
- Base model pairwise accuracy: 98.2%

📦 **Download:** [HuggingFace Dataset](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs)

---

## Evaluation

### Core Metrics

```bash
# Standard evaluation
python src/eval/eval_pairwise_orm.py \
  --checkpoint <checkpoint_path> \
  --test_path data/processed/orm_pairwise_test.jsonl
```

### Robustness Tests

```bash
# Anti-symmetry validation (label-swap test)
python src/eval/test_label_swap.py --checkpoint <checkpoint_path>

# Near-tie stress test
python src/eval/test_near_tie.py --checkpoint <checkpoint_path>

# Template-wise analysis
python src/eval/test_template_wise.py --checkpoint <checkpoint_path>
```

### Results Summary

**Length-Based Robustness:**

| Token Range | Accuracy | Pairs |
|-------------|----------|-------|
| 0–128 | 95.5% | 442 |
| 128–256 | 99.7% | 332 |
| 256+ | 96.1% | 458 |

**Anti-Symmetry Validation:**

| Metric | Value |
|--------|-------|
| Swapped Accuracy | 3.75% |
| Correlation (Original vs Swapped) | -0.998 |

**Near-Tie Calibration:**

| Margin Threshold | Accuracy |
|------------------|----------|
| \|Δ\| < 0.05 | 43% |
| \|Δ\| < 0.10 | 48% |
| \|Δ\| < 0.50 | 71% |

---

## Usage

### Scoring Reasoning Traces

```python
from src.eval.utils_load_orm import load_orm_model

# Load model
model, tokenizer = load_orm_model(
    checkpoint_path="runs/pairwise_orm/orm_pairwise_final.pt",
    base_model_path="facebook/opt-1.3b"
)

# Score a trace
def score_trace(trace: str) -> float:
    inputs = tokenizer(trace, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        score = model(**inputs).squeeze().item()
    return score

# Compare traces
score_a = score_trace("1. Calculate cost: $20/4 = $5\n2. Total: $5 × 10 = $50\nFinal Answer: $50")
score_b = score_trace("1. Assume linear growth\n2. Random calculation\nFinal Answer: $38")

print(f"Preferred: {'Trace A' if score_a > score_b else 'Trace B'}")
```

### Best-of-N Sampling

```python
def select_best(candidates: list[str]) -> str:
    scores = [score_trace(c) for c in candidates]
    return candidates[scores.index(max(scores))]
```

### Integration with Agentic Systems

```python
# Tree search pruning
def should_expand(trace: str, threshold: float = 0.0) -> bool:
    return score_trace(trace) > threshold

# Combine with Process Reward Models
def hybrid_score(trace: str, orm_model, prm_model, alpha: float = 0.5):
    orm_score = orm_model.score(trace)
    prm_score = prm_model.score_steps(trace).mean()
    return alpha * orm_score + (1 - alpha) * prm_score
```

---

## Project Structure

```
.
├── configs/
│   └── pairwise_orm.yaml          # Training configuration
├── data/
│   └── processed/                 # Pairwise datasets (JSONL)
├── src/
│   ├── data/
│   │   └── dataloader_pairwise_orm.py
│   ├── eval/
│   │   ├── eval_pairwise_orm.py   # Main evaluation script
│   │   ├── test_label_swap.py     # Anti-symmetry test
│   │   ├── test_near_tie.py       # Uncertainty calibration
│   │   └── utils_load_orm.py      # Model loading utilities
│   ├── losses/
│   │   └── pairwise_loss.py       # Logistic pairwise loss
│   ├── models/
│   │   └── orm_scorer.py          # ORM architecture
│   ├── training/
│   │   └── train_pairwise_orm.py  # Training loop
│   └── utils/
│       ├── config_loader.py
│       └── training_utils.py
├── tools/                         # Dataset construction utilities
├── runs/                          # Checkpoints and logs
├── DATASET_CARD.md
├── MODEL_CARD.md
└── requirements.txt
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mishra2025pairwise-orm,
  title={Stable Outcome Reward Modeling via Pairwise Preference Learning},
  author={Mishra, Aklesh},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Resources

| Resource | Link |
|----------|------|
| 📄 Paper | [ArXiv (submitted, under moderation)](https://arxiv.org) |
| 🤗 Model | [HuggingFace](https://huggingface.co/LossFunctionLover/pairwise-orm-model) |
| 🤗 Dataset | [HuggingFace](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs) |
| 🐦 Twitter | [@iminevitable10](https://x.com/iminevitable10) |
| 💻 GitHub | [Coder-12](https://github.com/Coder-12) |

---

## Contributing

We welcome contributions! Whether it's bug fixes, new features, documentation improvements, or research extensions, we'd love your help.

Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

**Areas we're especially interested in:**
- Multi-domain evaluation (code, science, multi-turn reasoning)
- Integration with other agentic frameworks
- Training efficiency improvements
- Multilingual support

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds upon foundational research in reward modeling and agentic reasoning:

- [MagiCore-Agentic](https://arxiv.org/abs/2409.12147) — Robust multi-step reasoning through agentic orchestration
- [Training Verifiers](https://arxiv.org/abs/2110.14168) — Math word problem verification
- [Process & Outcome Feedback](https://arxiv.org/abs/2211.14275) — Combining reward signals

---

## Contact

**Aklesh Mishra**  
📧 akleshmishra7@gmail.com  
🐦 [@iminevitable10](https://x.com/iminevitable10)  
💻 [github.com/Coder-12](https://github.com/Coder-12)

---

<div align="center">

**If you find this useful, consider giving it a ⭐!**

</div>
