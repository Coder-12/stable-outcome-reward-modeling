import json
import random
import argparse
from tqdm import tqdm

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def build_global_pairs(data, negatives_per_positive=4, seed=42):
    random.seed(seed)

    positives = [ex for ex in data if int(ex["orm_label"]) == 1]
    negatives = [ex for ex in data if int(ex["orm_label"]) == 0]

    assert len(positives) > 0
    assert len(negatives) > 0

    pairs = []

    for pos in tqdm(positives, desc="Building pairs"):
        sampled_negs = random.sample(
            negatives,
            k=min(negatives_per_positive, len(negatives))
        )

        for neg in sampled_negs:
            pairs.append({
                "chosen": pos["input_text"],
                "rejected": neg["input_text"],
                "meta": {
                    "chosen_chain_id": pos.get("chain_id"),
                    "rejected_chain_id": neg.get("chain_id"),
                    "chosen_label": 1,
                    "rejected_label": 0
                }
            })

    return pairs

def write_jsonl(path, data):
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    pairs = build_global_pairs(
        data,
        negatives_per_positive=args.negatives_per_positive
    )

    print(f"Built {len(pairs)} global pairwise examples")
    write_jsonl(args.output, pairs)

if __name__ == "__main__":
    main()