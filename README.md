<div align="center">

# Stable Outcome Reward Modeling via Pairwise Preference Learning

**A robust, reproducible framework for training Outcome Reward Models (ORMs) for agentic reasoning systems**

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue)](#)
[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/LossFunctionLover/pairwise-orm-model)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HuggingFace-blue)](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

---

## Overview

This repository provides the complete implementation for training and evaluating **Pairwise Outcome Reward Models (ORMs)** as described in our paper. The pairwise formulation prioritizes stability and reproducibility under minimal supervision, achieving **96.3% pairwise accuracy** with training convergence in just **800 steps (~10 minutes on a single GPU)**.

## AI Safety Context

This work is motivated by challenges in scalable oversight for increasingly capable language models. In agentic training loops (e.g., best-of-N sampling, tree search, iterative refinement), reward models act as control signals rather than static evaluators.

This project focuses on identifying and enforcing outcome reward model properties—stability, calibration, and robustness—that are necessary for safe deployment in such settings.

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
  --checkpoint runs/pairwise_orm/best_model.pt \
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
[Final Token (EOS) Pooling - attention-mask aware]
         ↓
[Lightweight Linear Head] ← Only these parameters are trained
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
| Base Model | facebook/opt-1.3b |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 2e-5 |
| LR Schedule | Linear warmup (50 steps) + constant |
| Batch Size | 8 pairs |
| Gradient Clipping | Max norm 1.0 |
| Training Steps | 800 |
| Warmup | 50 steps |
| Precision | FP16 |

**Loss Function:**

```python
L = -log(sigmoid(f(x_chosen) - f(x_rejected)))
```

---

## Dataset

The model is trained on the **ORM Pairwise Preference Pairs** dataset — a carefully curated collection of reasoning trace preferences.

| Split | Pairs | Negatives/Positive |
|-------|-------|--------------------|
| Train | 41,656 | 8 |
| Validation | 1,144 | 4 |
| Test | 1,232 | 4 |

### Data Format

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

### Quality Metrics (Source Pointwise Dataset)

- **Pearson correlation**: r = 0.87
- **Spearman correlation**: ρ = 0.83
- **Base model pairwise accuracy**: 98.2%

📦 **Download**: [HuggingFace Dataset](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs)

---

## Evaluation

### Core Metrics

```bash
python src/eval/eval_pairwise_orm.py \
  --checkpoint runs/pairwise_orm/best_model.pt \
  --base_model facebook/opt-1.3b \
  --test_path data/processed/orm_pairwise_test.jsonl
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
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Download the trained model weights
model_path = hf_hub_download(
    repo_id="LossFunctionLover/pairwise-orm-model",
    filename="pairwise_orm.pt"
)

# Load the base encoder (frozen during training)
base_model = AutoModel.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Load the trained scoring head weights
ckpt = torch.load(model_path, map_location="cpu")
state = ckpt["model_state"] if "model_state" in ckpt else ckpt

head_state = {
    k.replace("score.", ""): v
    for k, v in state.items()
    if k.startswith("score.")
}

assert set(head_state.keys()) == {"weight", "bias"}

# Initialize scoring head (single linear layer)
hidden_size = base_model.config.hidden_size
scoring_head = torch.nn.Linear(hidden_size, 1)
scoring_head.load_state_dict(head_state)

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.eval().to(device)
scoring_head.eval().to(device)

# Score a single reasoning trace
def score_trace(trace_text: str) -> float:
    """
    Compute scalar reward for a reasoning trace.
    Higher scores indicate better reasoning quality.
    """
    inputs = tokenizer(
        trace_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get base model embeddings
        encoder_outputs = base_model(**inputs)
        # Pool at actual sequence end (accounts for padding)
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        pooled = encoder_outputs.last_hidden_state[torch.arange(seq_lengths.size(0)), seq_lengths]
        # Get reward score
        score = scoring_head(pooled).squeeze(-1).cpu().item()
    
    return score

# Example: Compare two reasoning traces
trace_1 = """
1. Calculate the cost per item: $20 / 4 = $5
2. Calculate total for 10 items: $5 × 10 = $50
3. Apply 10% discount: $50 × 0.9 = $45

Final Answer: $45
"""

trace_2 = """
1. Assume linear growth incorrectly
2. Multiply by unrelated constant
3. Round result arbitrarily

Final Answer: $38
"""

score_1 = score_trace(trace_1)
score_2 = score_trace(trace_2)

print(f"Trace 1 score: {score_1:.3f}")
print(f"Trace 2 score: {score_2:.3f}")
print(f"Preferred trace: {'Trace 1' if score_1 > score_2 else 'Trace 2'}")
print(f"Confidence (margin): {abs(score_1 - score_2):.3f}")
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
```

---

## Project Structure

```
.
├── configs/
│   └── pairwise_orm.yaml
├── data/
│   └── processed/
├── src/
│   ├── data/
│   │   └── dataloader_pairwise_orm.py
│   ├── eval/
│   │   └── eval_pairwise_orm.py
│   ├── losses/
│   │   └── pairwise_loss.py
│   ├── models/
│   │   └── orm_scorer.py
│   ├── training/
│   │   └── train_pairwise_orm.py
│   └── utils/
│       └── config_loader.py
├── tools/
├── runs/
├── README.md
└── requirements.txt
```

---

## Citation

```bibtex
@article{mishra2026pairwise-orm,
  title={Stable Outcome Reward Modeling via Pairwise Preference Learning},
  author={Mishra, Aklesh},
  year={2026},
  note={Preprint}
}
```

---

## License

This project is licensed under the **Apache 2.0 License**. See `LICENSE` for details.

---

## Contact

**Aklesh Mishra**
- Email: akleshmishra7@gmail.com
- GitHub: [@Coder-12](https://github.com/Coder-12)
- HuggingFace: [@LossFunctionLover](https://huggingface.co/LossFunctionLover)

---

## Acknowledgments

This research builds upon months of dedicated work in preference learning and agentic reasoning systems. Special thanks to:

- The **MagiCore-Agentic** team for their inspiring work on multi-step agentic reasoning
- The broader ML community for foundational research in reward modeling and RLHF
- Contributors to open-source tools (Transformers, PyTorch) that made this work possible