import os
import sys
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader_pairwise_orm import PairwiseORMDataset
from src.models.orm_scorer import ORMScorer
from src.utils.config_loader import load_config
from src.losses.pairwise_loss import pairwise_logistic_loss


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_pairwise(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    margins = []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        for k in batch:
            batch[k] = batch[k].to(device)

        s_pos = model(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
        )
        s_neg = model(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
        )

        correct += (s_pos > s_neg).sum().item()
        total += s_pos.numel()
        margins.append((s_pos - s_neg).mean().item())

    acc = correct / max(1, total)
    margin = float(np.mean(margins)) if margins else 0.0
    return acc, margin


# ----------------------------
# Training
# ----------------------------
def train(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = cfg["device"]

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = ORMScorer(cfg["model"]["base_model"]).to(device)
    model = model.float()

    # freeze base
    if cfg["model"]["freeze_base"]:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # print(next(model.score.parameters()).dtype)
    # print(any(p.requires_grad for p in model.encoder.parameters()))
    # print(any(p.requires_grad for p in model.score.parameters()))

    assert any(p.requires_grad for p in model.score.parameters())
    assert not any(p.requires_grad for p in model.encoder.parameters())

    train_ds = PairwiseORMDataset(
        cfg["data"]["train_path"],
        tokenizer,
        cfg["data"]["max_length"],
    )
    val_ds = PairwiseORMDataset(
        cfg["data"]["val_path"],
        tokenizer,
        cfg["data"]["max_length"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["train_batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scaler = torch.amp.GradScaler("cuda")
    use_amp = (device == "cuda")

    total_steps = min(
        cfg["train"]["max_steps"],
        len(train_loader) * cfg["train"]["epochs"]
    )
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_acc = -1.0
    global_step = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                s_pos = model(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                )
                s_neg = model(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                )

                loss = pairwise_logistic_loss(s_pos, s_neg)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            with torch.no_grad():
                acc = (s_pos > s_neg).float().mean().item()
                margin = (s_pos - s_neg).mean().clamp(-10, 10).item()

            if global_step % cfg["train"]["log_every"] == 0:
                pbar.set_description(
                    f"step={global_step} loss={loss.item():.4f} "
                    f"acc={acc:.3f} margin={margin:.3f}"
                )

            global_step += 1

            if global_step >= cfg["train"]["max_steps"]:
                break

        # ----------------------------
        # Validation
        # ----------------------------
        val_acc, val_margin = evaluate_pairwise(
            model,
            val_loader,
            device,
            cfg.get("eval", {}).get("max_val_batches"),
        )

        print(
            f"[VAL] epoch={epoch+1} "
            f"pairwise_acc={val_acc:.4f} "
            f"margin={val_margin:.3f}"
        )

        # ----------------------------
        # Save best
        # ----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(cfg["train"]["save_dir"], "best_model.pt")
            torch.save(model.state_dict(), path)
            print(f"✅ Saved new best model (val_acc={val_acc:.4f})")

        if val_margin > cfg["eval"].get("early_stop_margin", 0.8):
            print("🛑 Early stop: margin saturated")
            break

        if global_step >= cfg["train"]["max_steps"]:
            break

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pairwise_orm.yaml",
        help="Path to training config"
    )
    args = parser.parse_args()
    if args.config:
        train(args.config)
    else:
        train("configs/pairwise_orm.yaml")