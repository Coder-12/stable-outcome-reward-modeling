import sys
import torch
import argparse
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.orm_scorer import ORMScorer
from src.utils.config_loader import load_config


def main(args):
    # Load config
    cfg = load_config(args.config)

    # Instantiate model (same as training)
    model = ORMScorer(cfg["model"]["base_model"])
    model = model.float()

    # Load trained weights
    print(f"Loading weights from: {args.best_model}")
    state_dict = torch.load(args.best_model, map_location="cpu")
    model.load_state_dict(state_dict)

    # Build final artifact
    final_ckpt = {
        "model_state": model.state_dict(),
        "config": cfg,
        "metrics": {
            "val_pairwise_acc": args.val_acc,
            "val_margin": args.val_margin,
            "max_steps": args.max_steps,
            "early_stop": args.early_stop,
        },
        "meta": {
            "task": "pairwise_orm",
            "training_type": "pairwise_logistic",
            "base_model": cfg["model"]["base_model"],
            "frozen_base": cfg["model"]["freeze_base"],
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(final_ckpt, args.output)

    print(f"✅ Final ORM checkpoint saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="orm_pairwise_final.pt")

    # metrics (explicit = no assumptions)
    parser.add_argument("--val_acc", type=float, required=True)
    parser.add_argument("--val_margin", type=float, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--early_stop", action="store_true")

    args = parser.parse_args()
    main(args)