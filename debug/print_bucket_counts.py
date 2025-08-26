from train.bucketer import BucketKey


def print_counts(counts):
    # pretty grid
    phases = ["early", "mid", "late"]
    outcomes = ["win", "draw", "loss"]
    print("\n[Bucket counts]")
    for o in outcomes:
        row = []
        for p in phases:
            row.append(str(counts.get(BucketKey(o, p), 0)).rjust(6))
        print(o.ljust(6), " ".join(row))
