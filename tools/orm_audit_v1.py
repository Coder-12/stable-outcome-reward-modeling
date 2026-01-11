import json
import statistics
import re

# --- CONFIGURATION ---
TRAIN_FILE = 'data/processed/orm_train_new.jsonl'
VAL_FILE = 'data/processed/orm_val_new.jsonl'


def extract_gold_number(gold_text):
    """Extracts the number after #### in the gold answer."""
    if not gold_text: return None
    match = re.search(r'####\s*(-?[\d\.,]+)', gold_text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            return None
    return None


def analyze_dataset():
    print(f"🔍 AUDITING: {TRAIN_FILE} vs {VAL_FILE}...\n")

    # 1. Store Data
    train_qids = set()
    val_qids = set()

    pos_lengths = []
    neg_lengths = []

    neg_samples = []  # To store (Gold, Final, Input, Generator)

    # --- PROCESS TRAINING SET ---
    try:
        with open(TRAIN_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)

                # A. Leakage Check (Store QIDs)
                train_qids.add(data.get('qid'))

                # B. Length Check (Simple word count)
                text = data.get('input_text', '')
                length = len(text.split())
                label = data.get('label')

                if label == 1:
                    pos_lengths.append(length)
                else:
                    neg_lengths.append(length)

                    # C. Negative Sample Collection (for Easy Negative check)
                    # We prioritize showing 'arith_corrupt' or synthetic errors
                    gen_source = data.get('meta', {}).get('generated_by', 'Unknown')
                    final = data.get('final_answer')
                    gold_text = data.get('meta', {}).get('gold_answer', '')
                    gold_num = extract_gold_number(gold_text)

                    if gold_num is not None and final:
                        try:
                            final_num = float(str(final).replace(',', ''))
                            diff = abs(gold_num - final_num)
                            neg_samples.append({
                                'gold': gold_num,
                                'bad': final_num,
                                'diff': diff,
                                'gen': gen_source,
                                'input': text[:100] + "..."  # First 100 chars
                            })
                        except:
                            pass  # Skip non-numeric finals for this specific calc
    except FileNotFoundError:
        print(f"❌ Error: Could not find {TRAIN_FILE}")
        return

    # --- PROCESS VAL SET ---
    try:
        with open(VAL_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                val_qids.add(data.get('qid'))
    except FileNotFoundError:
        print(f"❌ Error: Could not find {VAL_FILE}")
        return

    # --- GENERATE REPORT ---

    # 1. LEAKAGE REPORT
    overlaps = train_qids.intersection(val_qids)
    print("DATA LEAKAGE CHECK")
    print("==================")
    print(f"Train Unique QIDs: {len(train_qids)}")
    print(f"Val Unique QIDs:   {len(val_qids)}")
    print(f"Overlaps Found:    {len(overlaps)}")
    if len(overlaps) == 0:
        print("✅ Status: PASSED (Clean separation)")
    else:
        print("❌ Status: FAILED (You are training on test questions!)")
    print("\n")

    # 2. LENGTH BIAS REPORT
    print("LENGTH BIAS CHECK (Token Estimates)")
    print("===================================")
    if pos_lengths and neg_lengths:
        avg_pos = statistics.mean(pos_lengths)
        avg_neg = statistics.mean(neg_lengths)
        ratio = avg_pos / avg_neg if avg_neg > 0 else 0

        print(f"Avg Length (Correct): {avg_pos:.1f} words")
        print(f"Avg Length (Wrong):   {avg_neg:.1f} words")
        print(f"Ratio (Pos/Neg):      {ratio:.2f}")

        if 0.85 <= ratio <= 1.15:
            print("✅ Status: PASSED (Lengths are similar)")
        else:
            print("⚠️ Status: WARNING (Model might learn length shortcuts)")
    else:
        print("❌ Error: Missing positive or negative samples.")
    print("\n")

    # 3. EASY NEGATIVE AUDIT
    print("NEGATIVE SAMPLE AUDIT (Hardness Check)")
    print("======================================")
    # Sort by 'diff' to see range. We want to see if errors are subtle or huge.
    # Show 3 smallest diffs (Subtle) and 3 largest diffs (Obvious)
    if neg_samples:
        sorted_negs = sorted(neg_samples, key=lambda x: x['diff'])

        print("--- Top 3 SUBTLE Errors (Hardest Negatives) ---")
        for item in sorted_negs[:3]:
            print(f"Gold: {item['gold']} | Bad: {item['bad']} | Diff: {item['diff']:.4f}")
            print(f"   [Source: {item['gen']}]")

        print("\n--- Top 3 OBVIOUS Errors (Easiest Negatives) ---")
        for item in sorted_negs[-3:]:
            print(f"Gold: {item['gold']} | Bad: {item['bad']} | Diff: {item['diff']:.1f}")
            print(f"   [Source: {item['gen']}]")

        print("\n🤔 ANALYSIS:")
        if sorted_negs[0]['diff'] < 1.0:
            print("✅ You have 'Hard Negatives' (small errors). Good for robustness.")
        else:
            print("⚠️ All negatives seem to be huge errors. Model might get lazy.")
    else:
        print("❌ No numeric negatives found to analyze.")


if __name__ == "__main__":
    analyze_dataset()