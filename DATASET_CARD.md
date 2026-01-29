---
language: en
license: apache-2.0
tags:
- reward-model
- preference-learning
- agentic-reasoning
- outcome-reward-model
- pairwise-preference
datasets:
- LossFunctionLover/orm-pairwise-preference-pairs
metrics:
- accuracy
pipeline_tag: text-classification
model-index:
- name: pairwise-orm-model
  results:
  - task:
      type: preference-learning
      name: Pairwise Preference Ranking
    dataset:
      name: ORM Pairwise Preference Pairs
      type: LossFunctionLover/orm-pairwise-preference-pairs
    metrics:
    - type: accuracy
      value: 96.3
      name: Pairwise Accuracy
    - type: confidence_interval
      value: "[95.3%, 97.1%]"
      name: Bootstrap 90% CI (200 resamples over test pairs)
---

# Pairwise Outcome Reward Model (ORM)

<div align="center">

**A Robust Preference Learning Model for Agentic Reasoning Systems**

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs)

</div>

## 📋 Model Description

This is a **Pairwise Outcome Reward Model (ORM)** designed for agentic reasoning systems. The model learns to rank reasoning traces through relative preference judgments rather than absolute quality scores, achieving superior stability and reproducibility compared to traditional pointwise approaches.

**Key Achievements:**
- ✅ **96.3% pairwise accuracy** with tight confidence intervals [95.3%, 97.1%]
- ✅ **Stable training** in just 800 optimization steps (~10 minutes on single GPU)
- ✅ **Strong anti-symmetry** (swapped accuracy: 3.75%, correlation: -0.998)
- ✅ **Calibrated uncertainty** on near-tie cases
- ✅ **Length-robust** performance (95.5% - 99.7% across token ranges)
- ✅ **Frozen base model** architecture for reproducibility

## 🎯 Intended Use

This model is designed for:
- **Best-of-N sampling** in reasoning tasks
- **Candidate ranking** in agentic search and tree-based reasoning
- **Outcome-level feedback** in multi-step reasoning systems
- **Integration with Process Reward Models (PRMs)** for comprehensive evaluation
- **Agentic frameworks** like MagiCore-Agentic for robust decision-making

## 🏗️ Architecture

```
Input Text (Reasoning Trace)
    ↓
[Frozen Base LM Encoder]  ← Pre-trained, frozen during training
    ↓
[Final Non-Padding Token Pooling] ← Final-token pooling uses attention-mask–aware indexing to correctly handle left padding.
    ↓
[Lightweight Linear Head]    ← Only these parameters are trained
    ↓
Scalar Reward Score
```

**Design Philosophy:**
- **Frozen encoder**: Leverages pre-trained representations, reduces overfitting
- **Lightweight head**: <1M trainable parameters for stability
- **Minimal architecture**: Prioritizes reproducibility over complexity

## 📊 Training Details

### Dataset Construction

The model was trained on a carefully curated pairwise preference dataset derived from high-quality reasoning traces:

**Original Pointwise Dataset:**
- Train: 9,482 examples
- Validation: 524 examples  
- Test: 547 examples
- Labels: Binary (correct=1, incorrect=0)

**Quality Validation (Base Model Log-Probability Analysis):**
- Pearson correlation: **r = 0.87** (p < 1e-162)
- Spearman correlation: **ρ = 0.83** (p < 1e-134)
- Base model pairwise accuracy: **98.2%**
- Mean log-prob (positive): -2.17
- Mean log-prob (negative): -3.64

These metrics confirm strong signal separation in the base model, validating dataset quality before pairwise transformation.

**Pairwise Dataset Construction:**

The pointwise data was transformed into pairwise preferences using a global sampling strategy:

```python
# For each positive example, sample N negative examples
# Creates (chosen, rejected) pairs where chosen=correct, rejected=incorrect
```

**Dataset Statistics:**
- **Training pairs**: 41,656 (8 negatives per positive)
- **Validation pairs**: 1,144 (4 negatives per positive)
- **Test pairs**: 1,232 (4 negatives per positive)

Each pair contains:
- `chosen`: Correct reasoning trace (label=1)
- `rejected`: Incorrect reasoning trace (label=0)
- `meta`: Chain IDs and labels for traceability

**Curation Process:**
- ✅ **Weeks of iterative filtering and validation, including synthetic error construction and manual spot checks** on original dataset
- ✅ **Rigorous filtering** for correctness and reasoning quality
- ✅ **Balanced sampling** across reasoning patterns and lengths
- ✅ **Verified anti-symmetry** through base model analysis

### Training Configuration

**Hyperparameters:**
- **Base Model**: facebook/opt-1.3b
- **Trainable Parameters**: Scoring head only (~500K-1M params)
- **Optimizer**: AdamW
  - Learning rate: 2e-5
  - Betas: (0.9, 0.999)
  - Weight decay: 0.01
- **Learning Rate Schedule**: Linear warmup (50 steps) + constant
- **Batch Size**: 8 pairs
- **Gradient Clipping**: Max norm 1.0
- **Training Steps**: 800
- **Mixed Precision**: FP16
- **Hardware**: Single GPU (A100/V100-class or equivalent)
- **Training Time**: ~10 minutes

**Loss Function:**
```python
# Logistic pairwise ranking loss
L = -log(sigmoid(f(x_chosen) - f(x_rejected)))
```

## 🔬 Evaluation Results

### Main Performance (Test Set: 1,232 pairs)

| Metric | Value |
|--------|-------|
| **Pairwise Accuracy** | **96.3%** |
| Bootstrap 90% CI (200 resamples over test pairs) | [95.3%, 97.1%] |
| Mean Margin | 1.40 |
| Median Margin | 1.52 |
| Std Deviation | 1.12 |
| Incorrect/Tied Pairs | 3.7% |

### Length-Based Robustness

| Token Range | Accuracy | Sample Size |
|-------------|----------|-------------|
| 0-128 tokens | 95.5% | 442 pairs |
| 128-256 tokens | **99.7%** | 332 pairs |
| 256+ tokens | 96.1% | 458 pairs |

**Key Insight**: Model does not exploit length heuristics; benefits from additional context in medium-length range.

### Anti-Symmetry Validation (Label-Swap Test)

| Metric | Value | Expected |
|--------|-------|----------|
| Swapped Accuracy | 3.75% | ≈ error rate (→ 0) ✅ |
| Mean Swapped Margin | -1.40 | -1.40 ✅ |
| Pearson Correlation (Original vs Swapped) | -0.998 | ~-1.0 ✅ |

**Conclusion**: Model learns true preference ordering, not positional artifacts.

### Near-Tie Uncertainty Calibration

| Margin Threshold | Accuracy | Interpretation |
|------------------|----------|----------------|
| \|Δ\| < 0.05 | 43% | Low confidence → near chance |
| \|Δ\| < 0.10 | 48% | Uncertain region |
| \|Δ\| < 0.20 | 60% | Moderate confidence |
| \|Δ\| < 0.50 | 71% | Higher confidence |

**Key Insight**: Smooth calibration curve indicates well-calibrated uncertainty—critical for agentic systems that need to defer when uncertain.

## 💻 Usage

### Installation

```bash
pip install transformers torch huggingface_hub
```

### Basic Usage

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

### Batch Scoring for Best-of-N Sampling

```python
def rank_candidates(candidates: list[str], return_scores: bool = False):
    """
    Rank multiple candidate reasoning traces.
    
    Args:
        candidates: List of reasoning trace strings
        return_scores: If True, return (ranked_candidates, scores)
    
    Returns:
        Ranked list of candidates (best first)
    """
    scores = [score_trace(cand) for cand in candidates]
    
    # Sort by score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_candidates = [candidates[i] for i in ranked_indices]
    
    if return_scores:
        ranked_scores = [scores[i] for i in ranked_indices]
        return ranked_candidates, ranked_scores
    
    return ranked_candidates

# Example usage
candidates = [trace_1, trace_2, ...]  # Multiple traces for same problem
best_trace = rank_candidates(candidates)[0]
```

### Integration with Agentic Systems

```python
# Example: Use ORM for tree search pruning
def should_expand_node(reasoning_trace: str, threshold: float = 0.0) -> bool:
    """
    Decide whether to expand a reasoning node based on ORM score.
    """
    score = score_trace(reasoning_trace)
    return score > threshold

# Example: Combine with PRM for comprehensive evaluation
def hybrid_evaluation(trace: str, orm_model, prm_model):
    """
    Combine outcome-level (ORM) and process-level (PRM) rewards.
    """
    orm_score = score_trace(trace)  # Outcome quality
    prm_scores = prm_model.score_steps(trace)  # Step-level correctness
    
    # Weighted combination
    final_score = 0.5 * orm_score + 0.5 * prm_scores.mean()
    return final_score
```

## 🔗 Related Work & Citation

This work builds upon and complements:

- **MagiCore-Agentic** ([Liu et al., 2024](https://arxiv.org/abs/2409.12147)): Robust multi-step reasoning through agentic orchestration
- **Training Verifiers** ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)): Math word problem verification
- **Process & Outcome Feedback** ([Uesato et al., 2022](https://arxiv.org/abs/2211.14275)): Combining reward signals

### Citation

If you use this model in your research, please cite:

```bibtex
@article{mishra2026orm,
  title={Stable Outcome Reward Modeling via Pairwise Preference Learning},
  author={Mishra, Aklesh},
  journal={arXiv preprint},
  year={2026},
  note={Under review}
}
```

## 🔗 Resources

- 📄 **Paper**: Submitted to arXiv (under review)
- 💾 **Dataset**: [HuggingFace](https://huggingface.co/datasets/LossFunctionLover/orm-pairwise-preference-pairs)

## 📧 Contact

**Aklesh Mishra**
- Email: akleshmishra7@gmail.com
- Independent Researcher

## 📝 License

This model is released under the **Apache 2.0 License**.

## 🙏 Acknowledgments

This research builds upon months of dedicated work in preference learning and agentic reasoning systems. Special thanks to:

- The **MagiCore-Agentic** team for their inspiring work on multi-step agentic reasoning
- The broader ML community for foundational research in reward modeling and RLHF
- Contributors to open-source tools (Transformers, PyTorch) that made this work possible

## 📊 Model Performance Summary

```
╔════════════════════════════════════════════════════════════╗
║  Pairwise ORM - Key Metrics                                ║
╠════════════════════════════════════════════════════════════╣
║  Pairwise Accuracy:        96.3% [95.3%, 97.1%]            ║
║  Training Steps:           800 (~10 min on single GPU)     ║
║  Dataset Quality (r):      0.87 (Pearson)                  ║
║  Anti-symmetry:            -0.998 correlation              ║
║  Length Robustness:        95.5% - 99.7% across ranges     ║
║  Uncertainty Calibration:  Smooth degradation near ties    ║
╚════════════════════════════════════════════════════════════╝
```

---

**Last Updated**: January 22, 2026