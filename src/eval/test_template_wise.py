import sys, os, json, torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataloader_pairwise_orm import PairwiseORMDataset
from src.eval.utils_load_orm import load_orm, load_tokenizer


def extract_template(ex):
    meta = ex.get("meta", {})
    # Prefer explicit template if present
    if "template" in meta:
        return meta["template"]
    # Otherwise infer from chain_id prefix
    cid = meta.get("chosen_chain_id", "unknown")
    return cid.split("-")[0]  # e.g. synv1 / pos / regen


@torch.no_grad()
def template_wise_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_orm(args.checkpoint, args.base_model, device)
    tokenizer = load_tokenizer(args.base_model)

    # Load raw JSONL (need meta)
    raw = []
    with open(args.test_path) as f:
        for line in f:
            raw.append(json.loads(line))

    # Group indices by template
    groups = defaultdict(list)
    for i, ex in enumerate(raw):
        tpl = extract_template(ex)
        groups[tpl].append(i)

    print(f"\nFound {len(groups)} templates\n")

    # Dataset for scoring
    ds = PairwiseORMDataset(args.test_path, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_margins = []
    for batch in tqdm(loader, desc="Scoring"):
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.amp.autocast("cuda", enabled=(device=="cuda")):
            s_pos = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            s_neg = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

        all_margins.append((s_pos - s_neg).detach().cpu())

    margins = torch.cat(all_margins)

    # Per-template stats
    print("📊 TEMPLATE-WISE RESULTS")
    for tpl, idxs in groups.items():
        if len(idxs) < 20:
            continue
        m = margins[idxs]
        acc = (m > 0).float().mean().item()
        mean_margin = m.mean().item()
        print(f"{tpl:15s} | n={len(idxs):4d} | acc={acc:.3f} | mean_margin={mean_margin:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    args = p.parse_args()
    template_wise_eval(args)


if __name__ == "__main__":
    main()