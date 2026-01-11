import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader_pairwise_orm import PairwiseORMDataset
from src.models.orm_scorer import ORMScorer


# ----------------------------
# Utilities
# ----------------------------
@torch.no_grad()
def compute_all_margins(model, loader, device):
    model.eval()
    margins = []
    lengths = []

    for batch in tqdm(loader, desc="Forward pass"):
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            s_pos = model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
            )
            s_neg = model(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
            )

        diff = (s_pos - s_neg).detach().cpu()
        margins.append(diff)
        lengths.append(batch["chosen_attention_mask"].sum(dim=1).cpu())

    margins = torch.cat(margins)  # shape: (N,)
    lengths = torch.cat(lengths)  # shape: (N,)
    length_bucket_analysis(margins, lengths)
    return margins


def compute_metrics_from_margins(margins: torch.Tensor):
    margins_np = margins.numpy()

    acc = (margins_np > 0).mean()
    return {
        "pairwise_acc": float(acc),
        "margin_mean": float(margins_np.mean()),
        "margin_p10": float(np.percentile(margins_np, 10)),
        "margin_p50": float(np.percentile(margins_np, 50)),
        "margin_p90": float(np.percentile(margins_np, 90)),
        "num_pairs": int(len(margins_np)),
    }


def bootstrap_ci_from_margins(margins, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    margins_np = margins.numpy()
    n = len(margins_np)

    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs.append((margins_np[idx] > 0).mean())

    lo, med, hi = np.percentile(accs, [5, 50, 95])
    return {
        "acc_ci_5": float(lo),
        "acc_ci_50": float(med),
        "acc_ci_95": float(hi),
    }


def length_bucket_analysis(margins, lengths):
    buckets = [(0 ,128), (128 ,256), (256 ,512)]
    for lo, hi in buckets:
        idx = (lengths >= lo) & (lengths < hi)
        if idx.sum() < 50:
            continue
        print(
            f"[LEN {lo}-{hi}] "
            f"acc={(margins[idx] > 0).float().mean():.3f} "
            f"mean_margin={margins[idx].mean():.3f}"
        )

# ----------------------------
# Main
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Load model
    model = ORMScorer(args.base_model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # support both raw state_dict and wrapped dict
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Dataset
    test_ds = PairwiseORMDataset(
        args.test_path,
        tokenizer,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    start = time.time()
    margins = compute_all_margins(model, test_loader, device)
    print(f"Eval time: {time.time( ) -start:.1f}s | pairs/sec: {len(margins ) /(time.time( ) -start):.1f}")

    # Evaluation
    metrics = compute_metrics_from_margins(margins)

    # Bootstrap
    ci = bootstrap_ci_from_margins(margins, n_boot=200)

    results = {
        "test_metrics": metrics,
        "bootstrap_ci": ci,
    }

    print(json.dumps(results, indent=2))

    print("Fraction <= 0 margins:", float((margins <= 0).float().mean()))
    print("Max margin:", float(margins.max()))
    print("Min margin:", float(margins.min()))

    probs = torch.sigmoid(margins).float()
    print("Prob mean:", probs.mean().item())
    print(
        "Prob p10/p50/p90:", torch.quantile(probs, torch.tensor([0.1 ,0.5 ,0.9], device=probs.device))
    )

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "orm_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(margins, os.path.join(args.out_dir, "margins.pt"))

    torch.save(
        {
            "model_state": model.state_dict(),
            "base_model": args.base_model,
            "test_metrics": metrics,
            "bootstrap_ci": ci,
            "num_pairs": metrics["num_pairs"],
            "notes": "Pairwise-only ORM evaluation (single-pass margins)",
        },
        os.path.join(args.out_dir, "orm_pairwise_final.pt")
    )

    print("✅ ORM evaluation complete (FAST & CORRECT). Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", default="runs/pairwise_orm/eval_results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    main(args)