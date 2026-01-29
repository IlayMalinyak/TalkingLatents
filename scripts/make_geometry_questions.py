import json
import math
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def load_latents(path: Path):
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("NumPy is required for geometry question generation. Please install numpy.") from e
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D latent array, got shape {arr.shape}")
    return arr


def build_index(src: List[Dict[str, Any]], index_field: str = 'index') -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Return records with valid index and a mapping json_index -> row idx."""
    records = []
    mapping = {}
    for i, it in enumerate(src):
        if index_field not in it:
            continue
        try:
            jidx = int(it[index_field])
        except (TypeError, ValueError):
            continue
        records.append({
            'json_idx': jidx,
            'pos': i,
            'obsid': int(it.get('obsid', it.get('stellar_data', {}).get('obsid', -1))),
            'stellar_data': it.get('stellar_data', {}),
        })
        mapping[jidx] = jidx  # assume 1:1 row alignment w.r.t. latent rows
    return records, mapping


def order_candidates(np, latents, anchor_idx: int, cand_idxs: List[int]) -> List[int]:
    a = latents[anchor_idx]
    C = latents[cand_idxs]
    d2 = ((C - a[None, :]) ** 2).sum(axis=1)
    order = np.argsort(d2)
    return [cand_idxs[int(i)] for i in order]


def closer_of_two(np, latents, q_idx: int, y_idx: int, z_idx: int) -> int:
    q = latents[q_idx]
    dy = ((latents[y_idx] - q) ** 2).sum()
    dz = ((latents[z_idx] - q) ** 2).sum()
    return y_idx if dy <= dz else z_idx


def make_geometry_entries(
    src_items: List[Dict[str, Any]],
    latents_path: Path,
    index_field: str = 'index',
    k_choices=(3, 4, 5),
    seed: int = 0,
    frac_order_tasks: float = 0.7,
    progress_every: int = 1000,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    import numpy as np  # requires numpy

    latents = load_latents(latents_path)
    items, mapping = build_index(src_items, index_field=index_field)

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    total = len(items)
    print(f"Geometry Q/A over {total} items, latents shape={latents.shape}")
    t0 = time.time()
    processed = 0

    # Ensure we don't sample unavailable indices
    valid_rows = [it['json_idx'] for it in items if 0 <= it['json_idx'] < latents.shape[0]]
    valid_set = set(valid_rows)
    if not valid_rows:
        raise ValueError("No items with a valid latent row index found")

    for it in items:
        a_idx = it['json_idx']
        if a_idx not in valid_set:
            continue

        task_is_order = (rng.random() < frac_order_tasks)
        k = rng.choice(k_choices)

        if task_is_order:
            # Sample candidate set (distinct from anchor)
            pool = [r for r in valid_rows if r != a_idx]
            if len(pool) < k:
                continue
            rng.shuffle(pool)
            cand_rows = pool[:k]
            ordered_rows = order_candidates(np, latents, a_idx, cand_rows)
            cand_obsids = [int(src_items[r].get('obsid', src_items[r].get('stellar_data', {}).get('obsid', -1))) for r in cand_rows]
            ordered_obsids = [int(src_items[r].get('obsid', src_items[r].get('stellar_data', {}).get('obsid', -1))) for r in ordered_rows]

            q_text = (
                f"Order these OBSIDs by proximity to OBSID {it['obsid']}: "
                + ", ".join(str(x) for x in cand_obsids)
            )
            a_text = "Order: " + ", ".join(str(x) for x in ordered_obsids)
            qa_type = 'geometry_order'
        else:
            # Closer-to task with two references
            pool = [r for r in valid_rows if r != a_idx]
            if len(pool) < 2:
                continue
            rng.shuffle(pool)
            y_idx, z_idx = pool[0], pool[1]
            closer_idx = closer_of_two(np, latents, a_idx, y_idx, z_idx)
            y_obs = int(src_items[y_idx].get('obsid', src_items[y_idx].get('stellar_data', {}).get('obsid', -1)))
            z_obs = int(src_items[z_idx].get('obsid', src_items[z_idx].get('stellar_data', {}).get('obsid', -1)))
            closer_obs = y_obs if closer_idx == y_idx else z_obs
            q_text = f"Is OBSID {it['obsid']} closer to {y_obs} or {z_obs}?"
            a_text = f"CloserTo: {closer_obs}"
            qa_type = 'geometry_closer'

        entry = {
            'index': it['json_idx'],
            'obsid': it['obsid'],
            'description': json.dumps({'Question': q_text, 'Description': a_text}),
            'stellar_data': it['stellar_data'],
            'processing_timestamp': time.time(),
            'qa_type': qa_type,
        }
        out.append(entry)

        processed += 1
        if progress_every and processed % progress_every == 0:
            elapsed = time.time() - t0
            rate = processed / max(1e-9, elapsed)
            eta = (total - processed) / max(1e-9, rate)
            print(f"Processed {processed}/{total} ({processed/total:.1%}) | {rate:.1f} it/s | ETA {eta/60:.1f} min")

        if limit and processed >= limit:
            print(f"Stopping early at limit={limit}")
            break

    return out


def main():
    import argparse
    p = argparse.ArgumentParser("Build geometry/topology Q/A JSON from latents (no DR)")
    p.add_argument('--src', type=Path, default=Path('data/stellar_descriptions_questions_short.json'))
    p.add_argument('--latents', type=Path, required=True, help='Path to fm latent array (.npy) aligned by index')
    p.add_argument('--dst', type=Path, default=Path('data/stellar_descriptions_questions_geometry.json'))
    p.add_argument('--index_field', type=str, default='index')
    p.add_argument('--k_choices', type=str, default='3,4,5')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--frac_order_tasks', type=float, default=0.7)
    p.add_argument('--progress_every', type=int, default=1000)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    src = load_json(args.src)
    k_choices = tuple(int(x) for x in args.k_choices.split(',')) if args.k_choices else (3,)
    out = make_geometry_entries(
        src_items=src,
        latents_path=args.latents,
        index_field=args.index_field,
        k_choices=k_choices,
        seed=args.seed,
        frac_order_tasks=args.frac_order_tasks,
        progress_every=args.progress_every,
        limit=args.limit,
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open('w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} items to {args.dst}")


if __name__ == '__main__':
    main()

