import sys, os, torch, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataloader_pairwise_orm import PairwiseORMDataset
from src.eval.utils_load_orm import load_orm, load_tokenizer


@torch.no_grad()
def near_tie_test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_orm(args.checkpoint, args.base_model, device)
    tokenizer = load_tokenizer(args.base_model)

    ds = PairwiseORMDataset(args.test_path, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    margins = []
    for batch in tqdm(loader, desc="Scoring"):
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.amp.autocast("cuda", enabled=(device=="cuda")):
            s_pos = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            s_neg = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

        margins.append((s_pos - s_neg).detach().cpu())

    margins = torch.cat(margins)

    print("\n🎯 NEAR-TIE STRESS TEST")
    for eps in [0.05, 0.1, 0.2, 0.5]:
        idx = margins.abs() < eps
        if idx.sum() < 20:
            continue
        acc = (margins[idx] > 0).float().mean().item()
        print(f"|margin| < {eps:<4} | n={idx.sum():4d} | acc={acc:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    args = p.parse_args()
    near_tie_test(args)


if __name__ == "__main__":
    main()