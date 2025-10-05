import json
import math
import random
import time
from pathlib import Path

# Optional acceleration with NumPy if available
try:
    import numpy as np  # type: ignore
except Exception:
    np = None


def load_source(path: Path):
    with path.open('r') as f:
        return json.load(f)


def col_stats(mat):
    # mat: list of [x,y,z]
    n = len(mat)
    sums = [0.0, 0.0, 0.0]
    for a, b, c in mat:
        sums[0] += a; sums[1] += b; sums[2] += c
    mu = [s / max(1, n) for s in sums]
    var = [0.0, 0.0, 0.0]
    for a, b, c in mat:
        var[0] += (a - mu[0]) ** 2
        var[1] += (b - mu[1]) ** 2
        var[2] += (c - mu[2]) ** 2
    sd = [math.sqrt(v / max(1, n)) + 1e-8 for v in var]
    return mu, sd


TEMPLATES = [
    # approx values
    "Retrieve {k} stars with similar properties: Teff≈{teff_approx} K, log g≈{logg_approx}, [Fe/H]≈{feh_approx}. Return their OBSIDs as a comma-separated list.",
    "Find the top {k} nearest neighbors to a star with Teff≈{teff_approx} K, log g≈{logg_approx}, [Fe/H]≈{feh_approx}. List the OBSIDs only.",
    "Give {k} closest matches by latent similarity for Teff≈{teff_approx} K, log g≈{logg_approx}, [Fe/H]≈{feh_approx}. Output OBSIDs only.",
    "Return {k} catalog IDs (OBSIDs) most similar to Teff≈{teff_approx} K, log g≈{logg_approx}, [Fe/H]≈{feh_approx}.",
    # ranges
    "Identify {k} stars most similar to a population with Teff in {teff_rng} K, log g in {logg_rng}, and [Fe/H] in {feh_rng}. Provide their OBSIDs.",
    "Return {k} example stars matching Teff {teff_rng} K, log g {logg_rng}, [Fe/H] {feh_rng}. Output OBSIDs only.",
    "List {k} nearest neighbors for Teff within {teff_rng} K, log g in {logg_rng}, [Fe/H] in {feh_rng}. Give OBSIDs separated by commas.",
]

# Candidate ranking templates (present a candidate list; ask for the correct order)
RANK_TEMPLATES = [
    "Rank the following OBSIDs by similarity to Teff≈{teff_approx} K, log g≈{logg_approx}, [Fe/H]≈{feh_approx} (1=nearest): {candidates}. Provide the ordered list.",
    "Order these candidates from most to least similar given Teff {teff_rng} K, log g {logg_rng}, [Fe/H] {feh_rng}: {candidates}. Return OBSIDs in order.",
    "Given the target properties (Teff≈{teff_approx} K, log g≈{logg_approx}, [Fe/H]≈{feh_approx}), sort these OBSIDs by proximity: {candidates}. Output nearest-first order.",
]


def _format_props(teff: float, logg: float, feh: float):
    # approximations
    t_round = int(round(teff / 50.0) * 50)
    g_round = round(logg, 2)
    z_round = round(feh, 2)

    # ranges
    t_lo = int(round((teff - 150) / 50.0) * 50)
    t_hi = int(round((teff + 150) / 50.0) * 50)
    g_lo = round(logg - 0.15, 2)
    g_hi = round(logg + 0.15, 2)
    z_lo = round(feh - 0.10, 2)
    z_hi = round(feh + 0.10, 2)

    return {
        "teff_approx": t_round,
        "logg_approx": f"{g_round:.2f}",
        "feh_approx": f"{z_round:.2f}",
        "teff_rng": f"{t_lo}–{t_hi}",
        "logg_rng": f"{g_lo:.2f}–{g_hi:.2f}",
        "feh_rng": f"{z_lo:.2f}–{z_hi:.2f}",
    }


def build_retrieval_entries(
    src,
    k_choices=(1, 3, 5),
    seed: int = 0,
    rank_prob: float = 0.33,
    max_candidate_pool: int = 8,
    progress_every: int = 500,
    limit: int = 0,
):
    rng = random.Random(seed)
    # Extract features and ids
    obsids = []
    feats = []
    for item in src:
        sd = item.get('stellar_data', {})
        try:
            teff = float(sd.get('Teff'))
            logg = float(sd.get('logg'))
            feh = float(sd.get('FeH'))
        except (TypeError, ValueError):
            continue
        obsids.append(int(item.get('obsid', sd.get('obsid', -1))))
        feats.append([teff, logg, feh])

    if np is not None:
        F = np.asarray(feats, dtype=np.float32)
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True) + 1e-8
        Z = (F - mu) / sd  # [N,3]
    else:
        mu, sd = col_stats(feats)
        Z = [
            [ (f[0]-mu[0])/sd[0], (f[1]-mu[1])/sd[1], (f[2]-mu[2])/sd[2] ]
            for f in feats
        ]

    # Build index mapping obsid -> row
    id2row = {obsid: i for i, obsid in enumerate(obsids)}

    out = []
    now = time.time()
    total_items = len(src)
    print(f"Building retrieval entries: {total_items} source items; using numpy={bool(np)}")
    t0 = time.time()
    processed = 0

    for item in src:
        sd = item.get('stellar_data', {})
        try:
            teff = float(sd.get('Teff'))
            logg = float(sd.get('logg'))
            feh = float(sd.get('FeH'))
        except (TypeError, ValueError):
            continue
        obsid = int(item.get('obsid', sd.get('obsid', -1)))
        if obsid not in id2row:
            # skip malformed entries
            continue

        row = id2row[obsid]
        # squared L2 distance in standardized space
        if np is not None:
            q = Z[row]
            d2 = np.sum((Z - q[None, :]) ** 2, axis=1)
            d2[row] = np.inf
            # Determine a pool size to get both ranking and k neighbors efficiently
            pool = max(max_candidate_pool, max(k_choices))
            if pool >= d2.shape[0]:
                idx_sorted = np.argsort(d2)
            else:
                part = np.argpartition(d2, kth=pool-1)[:pool]
                part_sorted = part[np.argsort(d2[part])]
                idx_sorted = part_sorted
            dists = [(int(i), float(d2[int(i)])) for i in idx_sorted]
        else:
            q = Z[row]
            dists = []
            for i, z in enumerate(Z):
                if i == row:
                    dists.append((i, float('inf')))
                else:
                    d2 = (z[0]-q[0])**2 + (z[1]-q[1])**2 + (z[2]-q[2])**2
                    dists.append((i, d2))
            dists.sort(key=lambda t: t[1])
        k = rng.choice(k_choices)

        # Decide question mode: ranking or direct neighbor listing
        use_rank = (rng.random() < rank_prob)

        props = _format_props(teff, logg, feh)
        if use_rank:
            # Build a candidate set drawn from the top pool; then shuffle presentation
            pool = min(max_candidate_pool, len(dists))
            # ensure candidates >= k (use k or k+random)
            cand_size = max(k, min(pool, k + rng.randint(0, max(0, pool - k))))
            cand_indices = [i for i, _ in dists[:cand_size]]
            cand_ids = [int(obsids[i]) for i in cand_indices]
            # Correct order is nearest-first as per dists
            correct_order = cand_ids[:]
            # Shuffle presentation order
            presented = cand_ids[:]
            rng.shuffle(presented)
            candidates_str = ", ".join(str(x) for x in presented)
            tmpl = rng.choice(RANK_TEMPLATES)
            q_text = tmpl.format(k=k, candidates=candidates_str, **props)
            a_text = "Order: " + ", ".join(str(x) for x in correct_order[:k])
        else:
            # Direct neighbor IDs
            nn_obsids = [int(obsids[i]) for i, _ in dists[:k]]
            tmpl = rng.choice(TEMPLATES)
            q_text = tmpl.format(k=k, **props)
            a_text = "Neighbors: " + ", ".join(str(x) for x in nn_obsids)

        entry = {
            "index": item.get("index"),
            "obsid": obsid,
            "description": json.dumps({
                "Question": q_text,
                "Description": a_text,
            }),
            "stellar_data": item.get("stellar_data", {}),
            "processing_timestamp": now,
        }
        out.append(entry)

        processed += 1
        if progress_every > 0 and processed % progress_every == 0:
            elapsed = time.time() - t0
            rate = processed / max(1e-9, elapsed)
            remaining = (total_items - processed)
            eta = remaining / max(1e-9, rate)
            print(f"Processed {processed}/{total_items} ({processed/total_items:.1%}) | {rate:.1f} it/s | ETA {eta/60:.1f} min")

        if limit and processed >= limit:
            print(f"Stopping early at limit={limit}")
            break

    return out


def main():
    import argparse
    p = argparse.ArgumentParser("Build retrieval-style Q/A JSON from base file")
    p.add_argument('--src', type=Path, default=Path('data/stellar_descriptions_questions_short.json'))
    p.add_argument('--dst', type=Path, default=Path('data/stellar_descriptions_questions_retrieval.json'))
    p.add_argument('--k_choices', type=str, default='1,3,5', help='comma-separated list of k values to sample per item')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rank_prob', type=float, default=0.33, help='probability to emit ranking-style question')
    p.add_argument('--max_candidate_pool', type=int, default=8, help='max candidate list length for ranking questions')
    p.add_argument('--progress_every', type=int, default=500, help='print progress every N processed items (0 disables)')
    p.add_argument('--limit', type=int, default=0, help='process at most N items (0 = all)')
    args = p.parse_args()

    src = load_source(args.src)
    k_choices = tuple(int(x) for x in args.k_choices.split(',')) if args.k_choices else (3,)
    out = build_retrieval_entries(
        src,
        k_choices=k_choices,
        seed=args.seed,
        rank_prob=args.rank_prob,
        max_candidate_pool=args.max_candidate_pool,
        progress_every=args.progress_every,
        limit=args.limit,
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open('w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} entries to {args.dst}")


if __name__ == '__main__':
    main()
