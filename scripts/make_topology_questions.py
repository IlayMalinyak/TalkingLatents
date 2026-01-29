"""Generate topology-aware Q/A pairs for latent neighbourhood training.

This script replaces literal OBSID answers with physics-oriented summaries while
recording the true neighbour sets for retrieval supervision. It expects:

  * A JSON dataset where each item contains `index`, `obsid`, and
    `stellar_data` with Teff/logg/[Fe/H] values.
  * A NumPy `.npy` file (`features.npy`) holding latent embeddings aligned by
    dataframe index.

Outputs a JSON list where each entry has a Question/Description pair plus
metadata describing the anchor's nearest neighbours in latent space.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


PHYS_KEYS_DEFAULT = ("Teff", "logg", "FeH")


@dataclass
class DatasetEntry:
    index: int
    obsid: int
    stellar_data: Dict[str, float]


def load_dataset(path: Path) -> List[DatasetEntry]:
    raw = json.loads(path.read_text())
    entries: List[DatasetEntry] = []
    for item in raw:
        idx = item.get("index")
        obsid = item.get("obsid")
        if idx is None or obsid is None:
            continue
        entries.append(DatasetEntry(index=int(idx), obsid=int(obsid), stellar_data=item.get("stellar_data", {})))
    return entries


def extract_physics(entry: DatasetEntry, keys: Sequence[str]) -> np.ndarray:
    values = []
    for key in keys:
        val = entry.stellar_data.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            raise ValueError(f"Missing {key}")
        values.append(float(val))
    return np.asarray(values, dtype=np.float32)


def fmt_range(key: str, anchor: float, low: float, high: float) -> str:
    if key.lower().startswith("teff"):
        return f"Teff {low:.0f}–{high:.0f} K (anchor {anchor:.0f} K)"
    if key.lower().startswith("logg"):
        return f"log g {low:.2f}–{high:.2f} (anchor {anchor:.2f})"
    if key.lower().startswith("feh"):
        return f"[Fe/H] {low:.2f}–{high:.2f} (anchor {anchor:.2f})"
    return f"{key} {low:.3f}–{high:.3f} (anchor {anchor:.3f})"


def build_templates() -> Tuple[List[str], List[str]]:
    questions = [
        "Summarize the latent neighbourhood around this star (Teff={teff:.0f} K, log g={logg:.2f}, [Fe/H]={feh:.2f}). Include the expected parameter ranges for its {k} nearest neighbours.",
        "Given a stellar anchor with Teff={teff:.0f} K, log g={logg:.2f}, [Fe/H]={feh:.2f}, describe how tightly clustered its {k}-nearest latent neighbours are across these physical dimensions.",
        "Characterize the local latent topology: what Teff, log g, and [Fe/H] intervals do the closest {k} neighbours of this star occupy?"
    ]
    answers = [
        "The nearest {k} neighbours remain within {phys_ranges}. Their latent dispersion corresponds to an average distance of {mean_dist:.3f} (max {max_dist:.3f}).",
        "Neighbourhood summary: {phys_ranges}. This cluster spans ΔTeff≈{spread_teff:.0f} K, Δlog g≈{spread_logg:.2f}, Δ[Fe/H]≈{spread_feh:.2f}, indicating a {tightness} region in latent space (mean distance {mean_dist:.3f}).",
        "For the {k} closest neighbours we observe {phys_ranges}, with latent distances averaging {mean_dist:.3f} and the farthest still within {max_dist:.3f}; overall this reflects a {tightness} local topology."
    ]
    return questions, answers


ANCHOR_QUESTION_TEMPLATES = [
    "Describe this star based on its stellar parameters.",
    "Summarize the physical properties of this star.",
    "What do the stellar parameters indicate about this object?"
]


RETRIEVAL_QUESTION_TEMPLATES = [
    "Retrieve the {k} latent neighbours consistent with these ranges: {phys_ranges}. List their OBSIDs only.",
    "Which {k} stars lie closest in latent space while staying within {phys_ranges}? Provide the OBSIDs.",
    "Give the OBSIDs of the {k} most similar latent neighbours that fall inside {phys_ranges}."
]

CHAIN_QUESTION_TEMPLATES = [
    "Describe the evolution of stellar parameters along a latent walk of {steps} steps starting from this star.",
    "How do Teff, log g, and [Fe/H] change as we follow {steps} successive latent neighbours beginning here?",
    "Trace the latent sequence for {steps} hops from this star and summarise the physical trend."
]


def format_anchor_answer(entry: DatasetEntry, physics_keys: Sequence[str]) -> str:
    sd = entry.stellar_data
    parts: List[str] = []
    teff = sd.get("Teff")
    if teff is not None:
        parts.append(f"Teff ≈ {float(teff):.0f} K")
    logg = sd.get("logg")
    if logg is not None:
        parts.append(f"log g ≈ {float(logg):.2f}")
    feh = sd.get("FeH")
    if feh is not None:
        parts.append(f"[Fe/H] ≈ {float(feh):.2f}")

    subclass = sd.get("subclass") or sd.get("class")
    class_phrase = ''
    if subclass:
        class_phrase = f"It is classified as {str(subclass).strip()}-type."

    is_giant = sd.get("is_giant")
    evo_phrase = ''
    if isinstance(is_giant, bool):
        evo_phrase = "It is a giant star." if is_giant else "It is likely on or near the main sequence."

    metallicity_comment = ''
    if feh is not None:
        feh_val = float(feh)
        if feh_val > 0.15:
            metallicity_comment = "It is metal-rich relative to the Sun."
        elif feh_val < -0.15:
            metallicity_comment = "It is metal-poor compared to the Sun."

    gravity_comment = ''
    if logg is not None:
        logg_val = float(logg)
        if logg_val >= 4.0:
            gravity_comment = "Surface gravity values indicate a dwarf-like object."
        elif logg_val <= 3.0:
            gravity_comment = "Surface gravity suggests an evolved star." if not evo_phrase else ''

    summary = ", ".join(parts)
    extra_bits = " ".join(bit for bit in [class_phrase, metallicity_comment, gravity_comment, evo_phrase] if bit)
    return f"This star has {summary}. {extra_bits}".strip()


def build_latent_chain(
    start_entry: DatasetEntry,
    steps: int,
    nn_model: NearestNeighbors,
    index_map: Dict[int, DatasetEntry],
    physics_keys: Sequence[str],
    features: np.ndarray,
) -> List[DatasetEntry]:
    if steps <= 0:
        return [start_entry]

    chain = [start_entry]
    visited = {start_entry.index}
    current = start_entry

    for _ in range(steps):
        feature_vec = features[current.index].reshape(1, -1)
        distances, indices = nn_model.kneighbors(feature_vec, return_distance=True)
        next_entry = None
        for idx in indices[0]:
            if int(idx) in visited:
                continue
            candidate = index_map.get(int(idx))
            if candidate is None:
                continue
            try:
                extract_physics(candidate, physics_keys)
            except ValueError:
                continue
            next_entry = candidate
            break

        if next_entry is None:
            break

        chain.append(next_entry)
        visited.add(next_entry.index)
        current = next_entry

    return chain


def describe_chain(entries: List[DatasetEntry], physics_keys: Sequence[str]) -> str:
    segments = []
    prev_phys = None
    for step, entry in enumerate(entries):
        phys = extract_physics(entry, physics_keys)
        parts = [f"Step {step}: Teff {phys[0]:.0f} K"]
        if len(phys) > 1:
            parts.append(f"log g {phys[1]:.2f}")
        if len(phys) > 2:
            parts.append(f"[Fe/H] {phys[2]:.2f}")
        if prev_phys is not None:
            deltas = phys - prev_phys
            delta_parts = [f"ΔTeff {deltas[0]:+.0f}"]
            if len(deltas) > 1:
                delta_parts.append(f"Δlog g {deltas[1]:+.2f}")
            if len(deltas) > 2:
                delta_parts.append(f"Δ[Fe/H] {deltas[2]:+.2f}")
            parts.append("(" + ", ".join(delta_parts) + ")")
        segments.append("; ".join(parts))
        prev_phys = phys
    return " | ".join(segments)


def classify_tightness(mean_dist: float, max_dist: float) -> str:
    if max_dist < 0.5:
        return "very compact"
    if max_dist < 1.0:
        return "tight"
    if max_dist < 1.5:
        return "moderately broad"
    return "diffuse"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create topology-aware Q/A dataset")
    parser.add_argument("--src", type=Path, required=True, help="Input JSON with stellar entries")
    parser.add_argument("--features", type=Path, required=True, help="NumPy .npy file with latent features")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--metadata_out", type=Path, default=None, help="Optional metadata JSON for neighbour IDs")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbours to summarise")
    parser.add_argument("--max_items", type=int, default=0, help="Limit number of generated entries")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physics_keys", nargs="*", default=list(PHYS_KEYS_DEFAULT))
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--no_anchor_descriptions", action="store_true",
                        help="Skip generating per-star property description Q/A")
    parser.add_argument("--no_latent_retrieval", action="store_true",
                        help="Skip generating neighbour retrieval Q/A entries")
    parser.add_argument("--chain_steps", type=int, default=0,
                        help="Number of neighbour hops to include in latent chain (0 disables)")
    parser.add_argument("--no_chain_questions", action="store_true",
                        help="Skip generating latent chain Q/A entries")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    entries = load_dataset(args.src)
    rng.shuffle(entries)
    total_entries = len(entries)
    if args.max_items:
        entries = entries[: args.max_items]

    features = np.load(args.features)
    if features.ndim != 2:
        raise ValueError("features array must be 2-D")

    max_index = max(entry.index for entry in entries)
    if max_index >= features.shape[0]:
        raise ValueError("Feature array smaller than largest index in dataset")

    physics_keys = tuple(args.physics_keys) if args.physics_keys else PHYS_KEYS_DEFAULT

    # Build kNN (anchor + neighbours)
    k_neighbors = min(args.k + 1, features.shape[0])
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric=args.metric)
    nn.fit(features)

    q_templates, a_templates = build_templates()

    outputs: List[Dict[str, object]] = []
    meta_records: List[Dict[str, object]] = []

    index_map = {entry.index: entry for entry in entries}
    timestamp = time.time()

    for idx, entry in enumerate(entries, start=1):
        try:
            anchor_phys = extract_physics(entry, physics_keys)
        except ValueError:
            continue

        feature_vec = features[entry.index].reshape(1, -1)
        distances, indices = nn.kneighbors(feature_vec, return_distance=True)
        distances = distances[0]
        indices = indices[0]

        anchor_idx_mask = indices != entry.index
        neighbour_indices = indices[anchor_idx_mask][: args.k]
        neighbour_distances = distances[anchor_idx_mask][: args.k]
        if len(neighbour_indices) == 0:
            continue

        neighbor_records = []
        for neigh_idx, neigh_dist in zip(neighbour_indices, neighbour_distances):
            neigh_entry = index_map.get(int(neigh_idx))
            if neigh_entry is None:
                continue
            try:
                phys = extract_physics(neigh_entry, physics_keys)
            except ValueError:
                continue
            neighbor_records.append((neigh_entry, phys, float(neigh_dist)))

        if not neighbor_records:
            continue

        phys_array = np.stack([rec[1] for rec in neighbor_records], axis=0)
        dist_array = np.asarray([rec[2] for rec in neighbor_records], dtype=np.float32)
        k_eff = phys_array.shape[0]

        # Build physics range summaries
        range_parts = []
        physics_ranges_meta: Dict[str, Dict[str, float]] = {}
        spreads: Dict[str, float] = {}
        for dim, key in enumerate(physics_keys):
            low = float(np.min(phys_array[:, dim]))
            high = float(np.max(phys_array[:, dim]))
            physics_ranges_meta[key] = {"min": low, "max": high}
            spreads[key.lower()] = high - low
            range_parts.append(fmt_range(key, float(anchor_phys[dim]), low, high))

        phys_ranges_text = "; ".join(range_parts)

        mean_dist = float(dist_array.mean())
        max_dist = float(dist_array.max())
        tightness = classify_tightness(mean_dist, max_dist)

        def get_anchor_value(label: str, default: float = float(anchor_phys[0])) -> float:
            label = label.lower()
            for dim, key in enumerate(physics_keys):
                if key.lower().startswith(label):
                    return float(anchor_phys[dim])
            return default

        teff_val = get_anchor_value("teff")
        logg_val = get_anchor_value("logg", default=float(anchor_phys[min(len(anchor_phys)-1, 1)]))
        feh_val = get_anchor_value("feh", default=float(anchor_phys[min(len(anchor_phys)-1, 2)]))

        spread_teff = spreads.get("teff", 0.0)
        spread_logg = spreads.get("logg", 0.0)
        spread_feh = spreads.get("feh", 0.0)

        question_template = rng.choice(q_templates)
        question_text = question_template.format(teff=teff_val, logg=logg_val, feh=feh_val, k=k_eff)

        answer_template = rng.choice(a_templates)
        answer_text = answer_template.format(
            k=k_eff,
            phys_ranges=phys_ranges_text,
            mean_dist=mean_dist,
            max_dist=max_dist,
            spread_teff=spread_teff,
            spread_logg=spread_logg,
            spread_feh=spread_feh,
            tightness=tightness,
        )

        chain_entries = None
        if args.chain_steps > 0:
            chain_entries = build_latent_chain(entry, args.chain_steps, nn, index_map, physics_keys, features)
            if len(chain_entries) <= 1:
                chain_entries = None

        if not args.no_anchor_descriptions:
            anchor_question = rng.choice(ANCHOR_QUESTION_TEMPLATES)
            anchor_answer = format_anchor_answer(entry, physics_keys)
            outputs.append({
                "index": entry.index,
                "obsid": entry.obsid,
                "description": json.dumps({"Question": anchor_question, "Description": anchor_answer}),
                "stellar_data": entry.stellar_data,
                "qa_type": "anchor_description",
                "processing_timestamp": timestamp,
            })

        if (not args.no_latent_retrieval) and neighbor_records:
            retrieval_question = rng.choice(RETRIEVAL_QUESTION_TEMPLATES).format(
                k=k_eff,
                phys_ranges=phys_ranges_text,
            )
            retrieval_answer = "Matches: " + ", ".join(str(rec[0].obsid) for rec in neighbor_records)
            outputs.append({
                "index": entry.index,
                "obsid": entry.obsid,
                "description": json.dumps({"Question": retrieval_question, "Description": retrieval_answer}),
                "stellar_data": entry.stellar_data,
                "qa_type": "latent_retrieval",
                "processing_timestamp": timestamp,
            })

        outputs.append({
            "index": entry.index,
            "obsid": entry.obsid,
            "description": json.dumps({"Question": question_text, "Description": answer_text}),
            "stellar_data": entry.stellar_data,
            "qa_type": "topology_summary",
            "processing_timestamp": timestamp,
        })

        if (not args.no_chain_questions) and chain_entries is not None:
            chain_steps = len(chain_entries) - 1
            chain_question = rng.choice(CHAIN_QUESTION_TEMPLATES).format(steps=chain_steps)
            chain_answer = describe_chain(chain_entries, physics_keys)
            outputs.append({
                "index": entry.index,
                "obsid": entry.obsid,
                "description": json.dumps({"Question": chain_question, "Description": chain_answer}),
                "stellar_data": entry.stellar_data,
                "qa_type": "latent_chain",
                "processing_timestamp": timestamp,
            })

        meta_records.append({
            "index": entry.index,
            "obsid": entry.obsid,
            "neighbor_indices": [rec[0].index for rec in neighbor_records],
            "neighbor_obsids": [rec[0].obsid for rec in neighbor_records],
            "neighbor_distances": dist_array.tolist(),
            "physics_ranges": physics_ranges_meta,
            "mean_distance": mean_dist,
            "max_distance": max_dist,
            "chain_indices": [chain_entry.index for chain_entry in chain_entries] if chain_entries else [],
            "chain_obsids": [chain_entry.obsid for chain_entry in chain_entries] if chain_entries else [],
            "chain_length": len(chain_entries) if chain_entries else 0,
        })

        if idx % 500 == 0 or idx == total_entries:
            print(f"Processed {idx}/{total_entries} entries ({idx/total_entries:.1%})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outputs, indent=2))
    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(meta_records, indent=2))


if __name__ == "__main__":
    main()
