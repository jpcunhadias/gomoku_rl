import argparse
import json
from collections import Counter
from utils.paths import cycle_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True, help="Cycle number of the JSONL to analyze")
    args = parser.parse_args()

    paths = cycle_paths(args.cycle)
    log_path = paths["sp_log"]

    print(f"--- Checking Duplicates for Cycle {args.cycle} ---")
    print(f"Loading JSONL from: {log_path}")

    hashes = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "canon_hash" in record:
                        hashes.append(record["canon_hash"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"ERROR: JSONL file not found at {log_path}")
        return

    if not hashes:
        print("No records with 'canon_hash' found.")
        return

    total_records = len(hashes)
    unique_records = len(set(hashes))
    duplication_rate = 1 - (unique_records / total_records)

    print(f"\n[Check] Duplication Rate")
    print(f"  - Total records with hash: {total_records}")
    print(f"  - Unique hashes:           {unique_records}")
    print(f"  - Duplication rate:        {duplication_rate:.2%}")

    # Show the most common duplicates
    print("\nTop 5 most duplicated hashes:")
    for h, count in Counter(hashes).most_common(5):
        if count > 1:
            print(f"  - Hash {h}: {count} times")

if __name__ == "__main__":
    main()
