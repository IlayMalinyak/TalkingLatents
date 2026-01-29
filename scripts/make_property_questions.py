import json
import math
import random
import time
from pathlib import Path
from typing import List, Dict, Any


def load_source(path: Path):
    with path.open('r') as f:
        return json.load(f)


def valid_props(sd: Dict[str, Any]) -> bool:
    try:
        float(sd.get('Teff'))
        float(sd.get('logg'))
        float(sd.get('FeH'))
        return True
    except (TypeError, ValueError):
        return False


NUM_TEMPLATES = [
    "Return {k} OBSIDs of stars with Teff in {teff_rng} K, log g in {logg_rng}, and [Fe/H] in {feh_rng}.",
    "List {k} catalog IDs for stars matching Teff {teff_rng} K, log g {logg_rng}, [Fe/H] {feh_rng}.",
    "Find {k} stars whose parameters satisfy: Teff∈{teff_rng} K, log g∈{logg_rng}, [Fe/H]∈{feh_rng}. Output OBSIDs only.",
]

TYPE_TEMPLATES = [
    "Return {k} OBSIDs of {stype}-type stars.",
    "Find {k} stars classified as {stype}-type. Provide OBSIDs only.",
    "List {k} catalog IDs for {stype}-type stars.",
]


def format_ranges(teff: float, logg: float, feh: float, dT: int, dg: float, dz: float):
    t_lo = int(round((teff - dT) / 50.0) * 50)
    t_hi = int(round((teff + dT) / 50.0) * 50)
    g_lo = round(logg - dg, 2)
    g_hi = round(logg + dg, 2)
    z_lo = round(feh - dz, 2)
    z_hi = round(feh + dz, 2)
    return {
        'teff_rng': f"{t_lo}–{t_hi}",
        'logg_rng': f"{g_lo:.2f}–{g_hi:.2f}",
        'feh_rng': f"{z_lo:.2f}–{z_hi:.2f}",
    }


def subclass_letter(sd: Dict[str, Any]) -> str:
    sc = sd.get('subclass')
    if isinstance(sc, str) and sc:
        return sc.strip()[0].upper()
    return ''


def build_property_entries(
    src: List[Dict[str, Any]],
    k_choices=(1, 3, 5),
    seed: int = 0,
    frac_type_queries: float = 0.25,
    teff_delta: int = 150,
    logg_delta: float = 0.15,
    feh_delta: float = 0.10,
    progress_every: int = 1000,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    # Build a compact DB of candidates with metadata
    db = []
    for it in src:
        sd = it.get('stellar_data', {})
        if not valid_props(sd):
            continue
        try:
            rec = {
                'obsid': int(it.get('obsid', sd.get('obsid', -1))),
                'Teff': float(sd.get('Teff')),
                'logg': float(sd.get('logg')),
                'FeH': float(sd.get('FeH')),
                'stype': subclass_letter(sd),
                'stellar_data': sd,
                'index': it.get('index'),
            }
            db.append(rec)
        except (TypeError, ValueError):
            continue

    out = []
    total = len(db)
    print(f"Building property-based Q/A: {total} candidates | seed={seed}")
    t0 = time.time()
    processed = 0

    # Iterate over anchors; for each, synthesize a query and select matches by predicate
    for rec in db:
        teff = rec['Teff']
        logg = rec['logg']
        feh = rec['FeH']
        stype = rec['stype']
        obsid = rec['obsid']

        k = rng.choice(k_choices)

        use_type = (stype in {'O','B','A','F','G','K','M'} and rng.random() < frac_type_queries)
        if use_type:
            # Type query
            candidates = [d['obsid'] for d in db if d['stype'] == stype]
            # Remove duplicates and possibly the anchor first to diversify
            candidates = [x for x in candidates if x != obsid]
            if not candidates:
                continue
            rng.shuffle(candidates)
            selected = candidates[:k]
            tmpl = rng.choice(TYPE_TEMPLATES)
            q_text = tmpl.format(k=len(selected), stype=stype)
            a_text = "Matches: " + ", ".join(str(x) for x in selected)
        else:
            # Numeric property ranges
            props = format_ranges(teff, logg, feh, teff_delta, logg_delta, feh_delta)
            # Select by predicate (inclusive ranges)
            matches = []
            t_lo, t_hi = props['teff_rng'].split('–')
            g_lo, g_hi = props['logg_rng'].split('–')
            z_lo, z_hi = props['feh_rng'].split('–')
            t_lo, t_hi = int(t_lo), int(t_hi)
            g_lo, g_hi = float(g_lo), float(g_hi)
            z_lo, z_hi = float(z_lo), float(z_hi)

            for d in db:
                if (t_lo <= d['Teff'] <= t_hi) and (g_lo <= d['logg'] <= g_hi) and (z_lo <= d['FeH'] <= z_hi):
                    matches.append(d['obsid'])
            # Prefer diversity: drop anchor first, then sample
            matches = [x for x in matches if x != obsid]
            if not matches:
                continue
            rng.shuffle(matches)
            selected = matches[:k]
            tmpl = rng.choice(NUM_TEMPLATES)
            q_text = tmpl.format(k=len(selected), **props)
            a_text = "Matches: " + ", ".join(str(x) for x in selected)

        entry = {
            'index': rec['index'],
            'obsid': obsid,
            'description': json.dumps({
                'Question': q_text,
                'Description': a_text,
            }),
            'stellar_data': rec['stellar_data'],
            'processing_timestamp': time.time(),
            'qa_type': 'property_retrieval',
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
    p = argparse.ArgumentParser("Build property-based retrieval Q/A JSON (no kNN)")
    p.add_argument('--src', type=Path, default=Path('data/stellar_descriptions_questions_short.json'))
    p.add_argument('--dst', type=Path, default=Path('data/stellar_descriptions_questions_properties.json'))
    p.add_argument('--k_choices', type=str, default='1,3,5')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--frac_type_queries', type=float, default=0.25)
    p.add_argument('--teff_delta', type=int, default=150)
    p.add_argument('--logg_delta', type=float, default=0.15)
    p.add_argument('--feh_delta', type=float, default=0.10)
    p.add_argument('--progress_every', type=int, default=1000)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    src = load_source(args.src)
    k_choices = tuple(int(x) for x in args.k_choices.split(',')) if args.k_choices else (3,)
    out = build_property_entries(
        src,
        k_choices=k_choices,
        seed=args.seed,
        frac_type_queries=args.frac_type_queries,
        teff_delta=args.teff_delta,
        logg_delta=args.logg_delta,
        feh_delta=args.feh_delta,
        progress_every=args.progress_every,
        limit=args.limit,
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open('w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} items to {args.dst}")


if __name__ == '__main__':
    main()

