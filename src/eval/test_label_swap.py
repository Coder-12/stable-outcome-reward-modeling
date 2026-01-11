# src/eval/test_label_swap.py
import sys
import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader_pairwise_orm import PairwiseORMDataset
from src.eval.utils_load_orm import load_orm, load_tokenizer


@torch.no_grad()
def label_swap_test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer (exactly same as other evals)
    model = load_orm(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        device=device,
    )
    tokenizer = load_tokenizer(args.base_model)

    # Dataset
    ds = PairwiseORMDataset(
        args.test_path,
        tokenizer,
        max_length=args.max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    diffs = []

    for batch in tqdm(loader, desc="Label-swap eval"):
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            # 🔁 INTENTIONALLY SWAPPED
            s_pos = model(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
            )
            s_neg = model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
            )

        diffs.append((s_pos - s_neg).detach().cpu())

    diffs = torch.cat(diffs)

    acc = (diffs > 0).float().mean().item()
    mean_margin = diffs.mean().item()

    print("\n🔁 LABEL SWAP TEST (ANTI-SYMMETRY CHECK)")
    print(f"Pairs evaluated      : {len(diffs)}")
    print(f"Pairwise accuracy    : {acc:.4f}  (expected ≈ 0.50)")
    print(f"Mean swapped margin  : {mean_margin:.4f}  (expected ≈ 0.00)")

    return {
        "label_swap_acc": acc,
        "label_swap_mean_margin": mean_margin,
        "num_pairs": len(diffs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Label-swap sanity test for pairwise ORM"
    )
    parser.add_argument("--checkpoint", required=True, help="ORM checkpoint (.pt)")
    parser.add_argument("--base_model", required=True, help="Base LM path/name")
    parser.add_argument("--test_path", required=True, help="Pairwise test jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=6)

    args = parser.parse_args()
    label_swap_test(args)


if __name__ == "__main__":
    main()