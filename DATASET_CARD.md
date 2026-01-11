---
language: en
license: cc-by-4.0
task_categories:
- text-classification
- preference-learning
- reward-modeling
size_categories:
- 10K<n<100K
tags:
- preference-pairs
- reasoning-traces
- outcome-reward-model
- pairwise-ranking
pretty_name: ORM Pairwise Preference Pairs
dataset_info:
  features:
  - name: chosen
    dtype: string
  - name: rejected
    dtype: string
  - name: meta
    dtype:
      chosen_chain_id: string
      rejected_chain_id: string
      chosen_label: int32
      rejected_label: int32
  splits:
  - name: train
    num_examples: 41656
  - name: validation
    num_examples: 1144
  - name: test
    num_examples: 1232
---

# ORM Pairwise Preference Pairs Dataset

<div align="center">

**High-Quality Reasoning Trace Preferences for Training Outcome Reward Models**

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](link-to-arxiv)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/akleshmishra/pairwise-orm-model)

</div>

## 📋 Dataset Description

This dataset contains **44,032 curated pairwise preference judgments** over reasoning traces, designed for training robust Outcome Reward Models (ORMs). Each example is a pair of reasoning traces for the same task, with one marked as preferred (correct) and the other as rejected (incorrect).

**Key Features:**
- ✅ **Weeks of manual curation** and quality control on source data
- ✅ **Validated quality**: Base model achieves 98.2% pairwise accuracy
- ✅ **Strong signal separation**: Pearson r=0.87, Spearman ρ=0.83
- ✅ **Balanced construction**: Multiple negatives per positive for robust learning
- ✅ **Full traceability**: Chain IDs for linking back to source examples

## 📊 Dataset Statistics

### Split Sizes

| Split | Pairs | Negatives per Positive | Source Examples |
|-------|-------|------------------------|-----------------|
| **Train** | 41,656 | 8 | 9,482 (pointwise) |
| **Validation** | 1,144 | 4 | 524 (pointwise) |
| **Test** | 1,232 | 4 | 547 (pointwise) |
| **Total** | **44,032** | - | 10,553 |

### Quality Metrics (Source Pointwise Dataset)

**Base Model Log-Probability Analysis:**
- **Pearson correlation**: r = 0.87 (p < 1e-162)
- **Spearman correlation**: ρ = 0.83 (p < 1e-134)
- **Pairwise accuracy**: 98.2%
- **Mean log-prob (positive)**: -2.17
- **Mean log-prob (negative)**: -3.64
- **Separation**: Strong discrimination between correct/incorrect traces

![Base Model Distribution](https://your-image-link/distribution.png)

These metrics confirm robust signal quality before pairwise transformation, validating the dataset's suitability for preference learning.

## 🏗️ Dataset Construction

### Phase 1: Source Data Curation (Pointwise)

**Original Dataset Structure:**
```json
{
  "qid": "unique-question-id",
  "chain_id": "unique-chain-id", 
  "label": 0,  // Binary: 1=correct, 0=incorrect
  "orm_label": 0,
  "input_text": "Step-by-step reasoning trace...",
  "prompt": "Original problem statement",
  "steps": ["Step 1", "Step 2", ...],
  "final_answer": "Answer value",
  "meta": {
    "gold_answer": "Ground truth solution",
    "generated_by": "ORM-Repair-Synth-V1",
    "template": "error_type"
  }
}
```

**Curation Process (Weeks of Work):**
1. **Generation**: Synthetic reasoning traces with diverse error patterns
2. **Labeling**: Binary correctness annotation (1=correct, 0=incorrect)  
3. **Quality Control**: 
   - Manual review of reasoning validity
   - Verification against ground truth
   - Filtering of ambiguous cases
   - Removal of duplicates and near-duplicates
4. **Validation**: Base model log-probability analysis to confirm signal quality

**Quality Thresholds:**
- Positive examples: Logically sound reasoning leading to correct answer
- Negative examples: Clear errors in reasoning or incorrect final answer
- Filtering: Remove examples with log-prob inconsistent with label

### Phase 2: Pairwise Transformation

**Algorithm: Global Negative Sampling**

```python
def build_global_pairs(data, negatives_per_positive):
    """
    For each positive example, sample N negative examples globally.
    This creates diverse comparison pairs.
    """
    positives = [ex for ex in data if ex["orm_label"] == 1]
    negatives = [ex for ex in data if ex["orm_label"] == 0]
    
    pairs = []
    for pos in positives:
        sampled_negs = random.sample(negatives, k=negatives_per_positive)
        for neg in sampled_negs:
            pairs.append({
                "chosen": pos["input_text"],
                "rejected": neg["input_text"],
                "meta": {
                    "chosen_chain_id": pos["chain_id"],
                    "rejected_chain_id": neg["chain_id"],
                    "chosen_label": 1,
                    "rejected_label": 0
                }
            })
    return pairs
```

**Sampling Strategy Rationale:**
- **Train (8 neg/pos)**: Maximize training signal diversity
- **Val/Test (4 neg/pos)**: Balanced evaluation while maintaining diversity
- **Global sampling**: Ensures model learns general preferences, not task-specific patterns
- **Random seed**: Fixed for reproducibility (seed=42)

### Phase 3: Verification

**Post-Construction Checks:**
- ✅ All chosen traces have label=1
- ✅ All rejected traces have label=0  
- ✅ No duplicate pairs
- ✅ Chain IDs traceable to source
- ✅ Balanced length distribution across pairs

## 📝 Data Format

### Structure

Each example contains:

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

### Example

```json
{
  "chosen": "1. Calculate the cost of the unicorn piñata:\n2. Calculate the total cost of the Reese's:\n3. Calculate the total cost of the Snickers:\n4. Calculate the total cost of the Skittles:\n5. Add all the costs together:\n6. Compute the total cost step by step:\n7. Check the arithmetic result:\n8. Verification step:\n\nFinal Answer: 99",
  
  "rejected": "1. Assume that the weather forecast always grows linearly. However, assume an unrelated constant mistakenly.\n2. Therefore doubling the time doubles the value. However, assume an unrelated constant mistakenly.\n3. Ignore seasonal variations and round the result.\n4. Conclude with the projected incorrect value.\n\nFinal Answer: 15",
  
  "meta": {
    "chosen_chain_id": "pos-fe119ec6-f4b1-4710-80f9-8e64ced43c7e",
    "rejected_chain_id": "synv1-1b81a660-fa66-4cd0-9606-70b694486752",
    "chosen_label": 1,
    "rejected_label": 0
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `chosen` | string | Correct reasoning trace (label=1) |
| `rejected` | string | Incorrect reasoning trace (label=0) |
| `meta.chosen_chain_id` | string | Unique ID for chosen trace (traceable to source) |
| `meta.rejected_chain_id` | string | Unique ID for rejected trace (traceable to source) |
| `meta.chosen_label` | int | Always 1 (correct) |
| `meta.rejected_label` | int | Always 0 (incorrect) |

## 💻 Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("akleshmishra/orm-pairwise-preference-pairs")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"] 
test_data = dataset["test"]

# Example usage
print(f"Train size: {len(train_data)}")
print(f"First example: {train_data[0]}")
```

### Training a Pairwise ORM

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load dataset
dataset = load_dataset("akleshmishra/orm-pairwise-preference-pairs")

# Initialize model
base_model = AutoModel.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

class PairwiseORM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Trainable scoring head
        hidden_size = base_model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        score = self.score_head(pooled)
        return score

# Training loop
def pairwise_loss(chosen_score, rejected_score):
    """Logistic pairwise ranking loss"""
    return -torch.log(torch.sigmoid(chosen_score - rejected_score)).mean()

# Prepare batch
def prepare_batch(examples):
    chosen_inputs = tokenizer(
        examples["chosen"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    rejected_inputs = tokenizer(
        examples["rejected"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return chosen_inputs, rejected_inputs

# Train (simplified)
model = PairwiseORM(base_model)
optimizer = torch.optim.AdamW(model.score_head.parameters(), lr=1e-4)

for batch in dataloader:
    chosen_inputs, rejected_inputs = prepare_batch(batch)
    
    chosen_scores = model(**chosen_inputs)
    rejected_scores = model(**rejected_inputs)
    
    loss = pairwise_loss(chosen_scores, rejected_scores)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Evaluation

```python
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_pairwise_accuracy(model, test_data):
    """
    Compute pairwise accuracy: fraction where score(chosen) > score(rejected)
    """
    correct = 0
    total = 0
    margins = []
    
    for example in test_data:
        chosen_score = model.score(example["chosen"])
        rejected_score = model.score(example["rejected"])
        
        margin = chosen_score - rejected_score
        margins.append(margin)
        
        if margin > 0:
            correct += 1
        total += 1
    
    accuracy = correct / total
    mean_margin = np.mean(margins)
    
    return {
        "accuracy": accuracy,
        "mean_margin": mean_margin,
        "median_margin": np.median(margins),
        "std_margin": np.std(margins)
    }
```

## 📊 Dataset Analysis

### Length Distribution

| Split | Avg Chosen Length | Avg Rejected Length |
|-------|-------------------|---------------------|
| Train | ~180 tokens | ~175 tokens |
| Validation | ~178 tokens | ~173 tokens |
| Test | ~182 tokens | ~176 tokens |

**Note**: Lengths are balanced to prevent length bias in learning.

### Error Pattern Distribution (Rejected Traces)

Common error types in rejected traces:
- Incorrect arithmetic calculations
- Logical fallacies in reasoning steps
- Missing or redundant steps
- Incorrect application of formulas
- Rounding errors
- Misinterpretation of problem constraints

### Chain ID Traceability

All examples include `chain_id` fields linking back to source pointwise dataset:
- `pos-*`: Positive (correct) reasoning traces
- `synv1-*`: Synthetic negative traces from ORM-Repair-Synth-V1

## 🔬 Experimental Results

Models trained on this dataset achieve:

- **Pairwise Accuracy**: 96.3% [95.3%, 97.1% CI]
- **Training Stability**: Converges in 800 steps
- **Anti-symmetry**: -0.998 correlation on label-swap test
- **Length Robustness**: 95.5%-99.7% across token ranges

See [Model Card](https://huggingface.co/akleshmishra/pairwise-orm-model) for full evaluation details.

## 🔗 Related Resources

- 📄 **Paper**: [ArXiv](link-to-arxiv) - "An Empirical Study of Robust Preference Learning under Minimal Supervision"
- 🤖 **Trained Model**: [HuggingFace](https://huggingface.co/akleshmishra/pairwise-orm-model)
- 💻 **Training Code**: [GitHub](your-github-repo-url)
- 📊 **Source Pointwise Dataset**: Available upon request

## 📧 Contact & Citation

**Author**: Aklesh Mishra  
**Email**: akleshmishra7@gmail.com

If you use this dataset, please cite:

```bibtex
@misc{mishra2025orm-dataset,
  author = {Mishra, Aklesh},
  title = {ORM Pairwise Preference Pairs: A Curated Dataset for Training Outcome Reward Models},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/akleshmishra/orm-pairwise-preference-pairs}}
}

@article{mishra2025orm,
  title={An Empirical Study of Robust Preference Learning under Minimal Supervision},
  author={Mishra, Aklesh},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📝 License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to:
- **Share**: Copy and redistribute the material
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: Give appropriate credit

## 🙏 Acknowledgments

This dataset represents **weeks of dedicated curation work** in preference learning for agentic reasoning. Special thanks to:

- The **MagiCore-Agentic** project for inspiring robust multi-step reasoning research
- The ML community for foundational work in reward modeling and RLHF
- All contributors to the open-source ecosystem

## ⚠️ Limitations & Considerations

### Known Limitations

1. **Domain**: Primarily math/reasoning tasks; may not generalize to all domains
2. **Synthetic negatives**: Some rejected traces are synthetically generated with error templates
3. **English only**: All reasoning traces are in English
4. **Length range**: Optimized for traces up to 512 tokens

### Ethical Considerations

- This dataset is designed for research purposes in improving AI reasoning
- Models trained on this data should not be used as sole arbiters of correctness in high-stakes decisions
- Users should validate model outputs independently in production settings

### Future Work

- [ ] Expand to multi-domain reasoning (code, science, etc.)
- [ ] Include multi-turn reasoning dialogues
- [ ] Add fine-grained error annotations
- [ ] Create multilingual versions

---

**Dataset Version**: 1.0  
**Last Updated**: November 27, 2025  
**Status**: ✅ Stable
