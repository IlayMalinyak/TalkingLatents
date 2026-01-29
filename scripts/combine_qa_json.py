import json
import random
from pathlib import Path


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2)


def combine(
    orig_items,
    retr_items,
    max_items: int = 0,
    retr_ratio: float = 0.5,
    seed: int = 0,
):
    rng = random.Random(seed)

    # Tag items for potential sampling strategies
    for it in orig_items:
        it.setdefault('qa_type', 'original')
    for it in retr_items:
        it.setdefault('qa_type', 'retrieval')

    # Shuffle copies to avoid bias
    orig = orig_items[:]
    retr = retr_items[:]
    rng.shuffle(orig)
    rng.shuffle(retr)

    if max_items > 0:
        # Draw according to ratio
        n_retr = int(round(max_items * retr_ratio))
        n_orig = max_items - n_retr
        retr = retr[:min(n_retr, len(retr))]
        orig = orig[:min(n_orig, len(orig))]
        combined = orig + retr
        rng.shuffle(combined)
    else:
        # Concatenate all and shuffle
        combined = orig + retr
        rng.shuffle(combined)

    # Reindex indices (optional) to keep them unique and sequential
    for i, it in enumerate(combined):
        it['index'] = i

    return combined


def main():
    import argparse
    p = argparse.ArgumentParser("Combine original and retrieval Q/A JSONs into a single file")
    p.add_argument('--orig', type=Path, default=Path('data/stellar_descriptions_questions_short.json'))
    p.add_argument('--retr', type=Path, default=Path('data/stellar_descriptions_questions_retrieval.json'))
    p.add_argument('--out', type=Path, default=Path('data/stellar_descriptions_questions_combined.json'))
    p.add_argument('--max_items', type=int, default=0, help='cap total items (0=all)')
    p.add_argument('--retr_ratio', type=float, default=0.5, help='fraction of retrieval items if capping')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    orig = load_json(args.orig)
    retr = load_json(args.retr)
    combined = combine(orig, retr, max_items=args.max_items, retr_ratio=args.retr_ratio, seed=args.seed)
    save_json(args.out, combined)
    print(f"Wrote {len(combined)} items to {args.out}")


if __name__ == '__main__':
    main()

