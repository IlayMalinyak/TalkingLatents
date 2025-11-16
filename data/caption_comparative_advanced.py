import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


def _install_dependencies() -> None:
    """Ensure required libraries are available without reinstalling in worker processes."""
    if os.environ.get("TL_DEPS_READY") == "1":
        return

    print("Installing python packages: google-genai, kiauhoku", flush=True)
    subprocess.run(
        ["pip", "install", "--quiet", "google-genai", "kiauhoku"],
        check=True
    )
    os.environ["TL_DEPS_READY"] = "1"


try:
    from google import genai
except ImportError:
    _install_dependencies()
    from google import genai

try:
    import kiauhoku  # noqa: F401
except ImportError:
    _install_dependencies()

import numpy as np
import pandas as pd
from tqdm import tqdm

WORD_LIMIT = 50
MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 4
BASE_DELAY = 0.4

COMPARATIVE_FOCI = [
    "decide which star will exhaust its core hydrogen first and justify the conclusion",
    "identify which star evolves off the main sequence sooner and why",
    "predict which star reaches the subgiant branch earlier and describe the observational signature",
    "assess which star becomes a red giant first and how its temperature will respond",
    "evaluate which star will be more luminous after another billion years and explain the driver",
    "determine which star ends its life first and describe the evolutionary path it follows",
    "compare how metallicity shapes their evolutionary pace and future spectral appearance",
    "judge which star accumulates a degenerate helium core sooner and describe its implication",
    "explain which star loses mass faster as it approaches the giant phases",
    "state which star will ignite helium earlier and how its surface gravity will change"
]

_CLIENT: Optional[genai.Client] = None


def giant_cond(teff: float, logg: float) -> bool:
    """Return True when the star satisfies the dwarf criterion from Ciardi et al. (2011)."""
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4.0
    else:
        thresh = 5.2 - (2.8e-4 * teff)
    return logg >= thresh


def parse_gemini_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON content from plain text or code-fenced responses."""
    text = response_text.strip()
    patterns = [
        r"^```json\s*(.*?)\s*```$",
        r"^```\s*(.*?)\s*```$",
        r"^`(.*?)`$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            break

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def init_gemini_client() -> Optional[genai.Client]:
    """Initialise Gemini client, expecting the key in google_api.txt."""
    key_path = "/data/TalkingLatents/google_api.txt"
    if not os.path.exists(key_path):
        print("Warning: google_api.txt not found. Gemini client not initialised.")
        return None

    api_key = open(key_path, "r", encoding="utf-8").read().strip()
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        print(f"Failed to initialise Gemini client: {exc}")
        return None


def _ensure_client() -> genai.Client:
    """Return a cached Gemini client, creating it on first use."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = init_gemini_client()
    if _CLIENT is None:
        raise RuntimeError("Gemini client could not be initialised.")
    return _CLIENT


def make_rng(seed: int, identifier: Any) -> random.Random:
    """Create a deterministic RNG keyed by seed and identifier."""
    key = f"{seed}|{identifier}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    value = int(digest[:8], 16)
    return random.Random(value)


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b[\w\-\[\]/\.]+?\b", text))


def sanitise_value(value: Any, precision: int = 2) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.{precision}f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def determine_stage(teff: Any, logg: Any) -> Optional[str]:
    try:
        teff_val = float(teff)
        logg_val = float(logg)
    except (TypeError, ValueError):
        return None
    is_dwarf = giant_cond(teff_val, logg_val)
    return "likely main-sequence dwarf" if is_dwarf else "likely evolved subgiant/giant"


def format_star_block(title: str, items: Dict[str, Any]) -> str:
    if not items:
        return ""
    lines = [f"\n{title}:"]
    for key, value in items.items():
        if value is None:
            continue
        label = key.replace("_", " ").replace("-", " ")
        label = " ".join(word.capitalize() for word in label.split())
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def build_pair_payload(star_a: Dict[str, Any], star_b: Dict[str, Any]) -> str:
    """Prepare descriptive blocks for both stars and their contrasts."""
    block_a = {
        "Teff (K)": sanitise_value(star_a.get("Teff")),
        "log g (cgs)": sanitise_value(star_a.get("logg"), precision=2),
        "[Fe/H]": sanitise_value(star_a.get("FeH"), precision=2),
        "Mass (Msun)": sanitise_value(star_a.get("mass"), precision=2),
        "Age (Gyr)": sanitise_value(star_a.get("age"), precision=2),
        "Luminosity (Lsun)": sanitise_value(star_a.get("luminosity"), precision=2),
        "Stage Estimate": determine_stage(star_a.get("Teff"), star_a.get("logg")),
    }
    block_b = {
        "Teff (K)": sanitise_value(star_b.get("Teff")),
        "log g (cgs)": sanitise_value(star_b.get("logg"), precision=2),
        "[Fe/H]": sanitise_value(star_b.get("FeH"), precision=2),
        "Mass (Msun)": sanitise_value(star_b.get("mass"), precision=2),
        "Age (Gyr)": sanitise_value(star_b.get("age"), precision=2),
        "Luminosity (Lsun)": sanitise_value(star_b.get("luminosity"), precision=2),
        "Stage Estimate": determine_stage(star_b.get("Teff"), star_b.get("logg")),
    }

    deltas = {}
    for label in ["Teff", "logg", "FeH", "mass", "age", "luminosity"]:
        val_a = star_a.get(label)
        val_b = star_b.get(label)
        try:
            val_a_f = float(val_a)
            val_b_f = float(val_b)
            delta = val_a_f - val_b_f
            if abs(delta) < 1e-6:
                continue
            if abs(delta) >= 100:
                deltas[f"Delta {label} (A-B)"] = f"{delta:.0f}"
            else:
                deltas[f"Delta {label} (A-B)"] = f"{delta:.2f}"
        except (TypeError, ValueError):
            continue

    payload = ""
    payload += format_star_block("Star A", block_a)
    payload += format_star_block("Star B", block_b)
    payload += format_star_block("Direct Comparisons", deltas)
    return payload


def create_comparative_prompt(star_a: Dict[str, Any],
                              star_b: Dict[str, Any],
                              focus: str,
                              feedback: str = "") -> str:
    """Build the prompt for Gemini for comparative QA."""
    prompt = """
You are an astrophysicist constructing advanced two-star comparison questions.
Every answer must remain scientifically correct and consistent with the provided data.
Create exactly one question and one answer satisfying these criteria:
1. Write the question as two sentences labelled "Part 1:" and "Part 2:". Part 1 must request predictions for the effective temperature (Teff), surface gravity (log g), and metallicity ([Fe/H]) of Star A and Star B without revealing numeric values. Feel free to vary the language (e.g., "effective temperature" or "stellar temperature"). Part 2 must pose an open comparative query addressing: {focus}.
2. Do not copy any actual numeric values into the question text. Use qualitative expressions instead of providing numbers.
3. Write the answer as two sentences labelled "Part 1:" and "Part 2:". Part 1 must present the requested parameters for Star A first and then Star B using concise, scientifically clear phrasing (abbreviations like 'Teff' are welcome but not mandatory). Part 2 must deliver the comparative evolutionary reasoning.
4. Refer explicitly to "Star A" and "Star B" in both the question and the answer, but otherwise aim for varied sentence structures and vocabulary between examples.
5. Keep both the question and the answer within {word_limit} words each; choose wording that remains precise yet compact.
6. Ensure the dataset stays diverse by mixing synonyms, alternate sentence structures, and different comparative angles.

Return only a JSON object using this schema:
{{
  "Question": "<question text>",
  "Answer": "<answer text>"
}}

Do not include markdown fences or additional commentary. Verify your word counts before returning the JSON.
"""

    formatted_prompt = prompt.format(focus=focus, word_limit=WORD_LIMIT)
    if feedback:
        formatted_prompt += f"\nPrevious issues to correct: {feedback.strip()}\n"

    payload = build_pair_payload(star_a, star_b)
    return formatted_prompt + payload


def validate_response(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return "Response was not a JSON object."

    question = data.get("Question")
    answer = data.get("Answer")
    if not question or not isinstance(question, str):
        return "Missing or invalid 'Question'."
    if not answer or not isinstance(answer, str):
        return "Missing or invalid 'Answer'."

    if "Part 1:" not in question or "Part 2:" not in question:
        return "Question must contain 'Part 1:' and 'Part 2:' segments."

    if "Part 1:" not in answer or "Part 2:" not in answer:
        return "Answer must contain 'Part 1:' and 'Part 2:' segments."

    q_words = count_words(question)
    a_words = count_words(answer)
    if q_words > WORD_LIMIT + 20 or a_words > WORD_LIMIT + 20:
        return (f"Word limit exceeded (Question: {q_words}, Answer: {a_words}). "
                f"Both must be ≤ {WORD_LIMIT}.")

    return None


def convert_to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    return obj


def preprocess_star_dict(star_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert normalised stellar quantities (e.g., Teff) into physical units."""
    data = dict(star_dict)
    teff = data.get("Teff")
    try:
        teff_val = float(teff)
        if teff_val < 200:
            data["Teff"] = teff_val * 5778.0
        else:
            data["Teff"] = teff_val
    except (TypeError, ValueError):
        pass
    return data


def select_focus(rng: random.Random) -> str:
    return rng.choice(COMPARATIVE_FOCI)


def generate_pair_entry(star_a: Dict[str, Any],
                        star_b: Dict[str, Any],
                        idx_a: Any,
                        idx_b: Any,
                        seed: int,
                        max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """Generate an advanced comparative QA pair for two stars."""
    client = _ensure_client()
    working_a = dict(star_a)
    working_b = dict(star_b)

    feedback = ""
    rng = make_rng(seed, f"{idx_a}|{idx_b}")

    for attempt in range(1, max_retries + 1):
        focus = select_focus(rng)
        prompt = create_comparative_prompt(working_a, working_b, focus, feedback)
        try:
            response = client.models.generate_content(
                contents=[prompt],
                model=MODEL_NAME,
                config={"response_mime_type": "application/json"}
            )
            response_text = getattr(response, "text", None)
            if not response_text:
                feedback = "Empty response. Provide the JSON object."
                time.sleep(BASE_DELAY * attempt)
                continue

            parsed = parse_gemini_json_response(response_text)
            error = validate_response(parsed)
            if error:
                feedback = error
                time.sleep(BASE_DELAY * attempt)
                continue

            index_map = {
                "a": int(idx_a) if isinstance(idx_a, (int, np.integer)) else idx_a,
                "b": int(idx_b) if isinstance(idx_b, (int, np.integer)) else idx_b,
            }
            return {
                "indices": index_map,
                "question": parsed["Question"].strip(),
                "answer": parsed["Answer"].strip(),
                "word_counts": {
                    "question": count_words(parsed["Question"]),
                    "answer": count_words(parsed["Answer"]),
                },
                "star_a": convert_to_native(working_a),
                "star_b": convert_to_native(working_b),
            }

        except Exception as exc:
            feedback = f"Exception encountered: {exc}"
            time.sleep(BASE_DELAY * attempt)

    print(f"Failed to generate comparative QA for indices {idx_a} and {idx_b}: {feedback}")
    return None


def _comparative_worker(args: Tuple[Any, Any, Dict[str, Any], Dict[str, Any], int, int]) -> Optional[Dict[str, Any]]:
    idx_a, idx_b, data_a, data_b, seed, max_retries = args
    return generate_pair_entry(data_a, data_b, idx_a, idx_b, seed, max_retries)


def _worker_initializer() -> None:
    """Reset cached resources for each worker process."""
    global _CLIENT
    _CLIENT = None


def _is_pair_diverse(feature_map: Dict[str, np.ndarray],
                     pos_a: int,
                     pos_b: int) -> bool:
    """
    Check whether two positions differ enough to form an informative pair.
    Returns True when at least one metric shows a meaningful difference.
    """
    thresholds = {
        "Teff": 120.0,
        "logg": 0.1,
        "FeH": 0.05,
        "mass": 0.05,
        "age": 0.2,
        "luminosity": 0.25,
    }

    evaluated = False
    for key, minimum in thresholds.items():
        values = feature_map.get(key)
        if values is None:
            continue
        evaluated = True
        val_a = values[pos_a]
        val_b = values[pos_b]
        if np.isnan(val_a) or np.isnan(val_b):
            continue
        if abs(val_a - val_b) >= minimum:
            return True
    return not evaluated


def build_pairs(df: pd.DataFrame,
                limit: Optional[int],
                rng: random.Random) -> List[Tuple[int, int]]:
    """
    Create star index pairs efficiently, preferring diverse combinations while avoiding
    expensive repeated DataFrame lookups. The algorithm shuffles indices once and then
    greedily forms pairs, scanning a small window when it needs to find a more diverse
    partner.
    """
    if len(df) < 2:
        return []

    available_indices = list(df.index)
    rng.shuffle(available_indices)

    max_pairs = len(available_indices) // 2
    if limit is not None:
        max_pairs = min(max_pairs, limit)
    if max_pairs == 0:
        return []

    usable_count = max_pairs * 2
    trimmed_indices = available_indices[:usable_count]

    subset = df.loc[trimmed_indices].reset_index()
    original_indices = subset["index"].tolist()

    numeric_columns = ["Teff", "logg", "FeH", "mass", "age", "luminosity"]
    feature_map: Dict[str, np.ndarray] = {}
    for column in numeric_columns:
        series_raw = subset.get(column)
        if series_raw is None:
            continue
        series = pd.to_numeric(series_raw, errors="coerce")
        if series.isnull().all():
            continue
        feature_map[column] = series.to_numpy(dtype=float)

    pairs: List[Tuple[int, int]] = []
    window = 32  # limit search radius to keep near-linear complexity

    pos = 0
    total_positions = len(original_indices)
    while pos + 1 < total_positions and len(pairs) < max_pairs:
        partner_pos = None
        upper_bound = min(total_positions, pos + 1 + window)
        for candidate_pos in range(pos + 1, upper_bound):
            if _is_pair_diverse(feature_map, pos, candidate_pos):
                partner_pos = candidate_pos
                break

        if partner_pos is None:
            partner_pos = pos + 1

        if partner_pos != pos + 1:
            original_indices[pos + 1], original_indices[partner_pos] = (
                original_indices[partner_pos],
                original_indices[pos + 1],
            )
            for array in feature_map.values():
                array[pos + 1], array[partner_pos] = array[partner_pos], array[pos + 1]

        pairs.append((original_indices[pos], original_indices[pos + 1]))
        pos += 2

    return pairs


def generate_dataset(input_csv: str,
                     output_json: str,
                     limit: Optional[int],
                     seed: int,
                     max_retries: int,
                     processes: int) -> None:
    df = pd.read_csv(input_csv)

    if len(df) < 2:
        raise ValueError("Need at least two stars to form comparative questions.")

    if processes < 1:
        raise ValueError("Number of processes must be at least 1.")

    rng = random.Random(seed)
    pair_limit = limit
    pairs = build_pairs(df, pair_limit, rng)

    print(f"Generated {len(pairs)} star pairs from {input_csv}")

    if not pairs:
        print("No pairs generated; nothing to do.")
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump([], fh, indent=2, ensure_ascii=False)
        return

    unique_indices = {idx for pair in pairs for idx in pair}
    row_cache: Dict[Any, Dict[str, Any]] = {}
    for idx in unique_indices:
        row_cache[idx] = preprocess_star_dict(convert_to_native(df.loc[idx].to_dict()))

    tasks: List[Tuple[Any, Any, Dict[str, Any], Dict[str, Any], int, int]] = [
        (idx_a, idx_b, row_cache[idx_a], row_cache[idx_b], seed, max_retries)
        for idx_a, idx_b in pairs
    ]

    results: List[Dict[str, Any]] = []

    if processes > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes, initializer=_worker_initializer) as pool:
            iterator = pool.imap_unordered(_comparative_worker, tasks)
            for entry in tqdm(iterator, total=len(tasks), desc="Generating comparative QA"):
                if entry:
                    results.append(entry)
    else:
        _worker_initializer()
        for task in tqdm(tasks, total=len(tasks), desc="Generating comparative QA"):
            entry = _comparative_worker(task)
            if entry:
                results.append(entry)

    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} comparative entries to {output_json}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate advanced comparative stellar evolution QA pairs using Gemini."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing stellar parameters.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination JSON file for the generated comparative dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of star pairs to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed controlling pair selection and diversity focus.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=MAX_RETRIES,
        help="Maximum number of attempts per star pair.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes to use (set to 1 to disable multiprocessing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    generate_dataset(
        input_csv=args.input,
        output_json=args.output,
        limit=args.limit,
        seed=args.seed,
        max_retries=args.retries,
        processes=args.processes,
    )
