import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
os.system('pip install google-genai kiauhoku')


import numpy as np
import pandas as pd
from google import genai
from tqdm import tqdm

WORD_LIMIT = 50
MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 4
BASE_DELAY = 0.4

EVOLUTIONARY_FOCI = [
    "forecast how its luminosity and radius drift over the next few hundred million years",
    "state which evolutionary phase it approaches after its core hydrogen is mostly exhausted",
    "describe its observable appearance once it begins shell hydrogen burning",
    "explain how its surface temperature and colour will shift as it climbs the subgiant branch",
    "summarise the changes expected as it ascends the red giant branch and when that happens",
    "determine whether it develops a degenerate helium core and how that shapes its future",
    "project its evolutionary state and photometric behaviour one gigayear from now",
    "clarify how rotation or metallicity influence its remaining main-sequence lifetime",
    "estimate when helium ignition occurs and what the star looks like at that stage",
    "discuss how mass loss may progress as it nears the asymptotic giant branch"
]

_CLIENT: Optional[genai.Client] = None


def giant_cond(teff: float, logg: float) -> bool:
    """
    Condition adopted from Ciardi et al. (2011) to separate dwarfs and giants.
    Returns True when the star satisfies the dwarf criterion.
    """
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4.0
    else:
        thresh = 5.2 - (2.8e-4 * teff)
    return logg >= thresh


def parse_gemini_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Handle JSON that might be wrapped in Markdown code fences."""
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
    """Initialise the Gemini client using google_api.txt."""
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
    """Return a cached Gemini client, initialising it when needed."""
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
    """Count words in a string, treating numbers and scientific notation as words."""
    if not text:
        return 0
    return len(re.findall(r"\b[\w\-\[\]/\.]+?\b", text))


def sanitise_value(value: Any, precision: int = 2) -> Optional[str]:
    """Render a value as a compact string or return None if it is missing."""
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
    """Provide a qualitative stage estimate from Teff and log g."""
    try:
        teff_val = float(teff)
        logg_val = float(logg)
    except (TypeError, ValueError):
        return None

    is_dwarf = giant_cond(teff_val, logg_val)
    return "likely main-sequence dwarf" if is_dwarf else "likely evolved subgiant/giant"


def format_prompt_block(title: str, items: Dict[str, Any]) -> str:
    """Format a section of the prompt."""
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


def build_star_prompt_payload(star_data: Dict[str, Any]) -> str:
    """Collect key stellar attributes for the prompt."""
    stellar_params = {
        "Teff (K)": sanitise_value(star_data.get("Teff")),
        "log g (cgs)": sanitise_value(star_data.get("logg"), precision=2),
        "[Fe/H]": sanitise_value(star_data.get("FeH"), precision=2),
        "Mass (Msun)": sanitise_value(star_data.get("mass"), precision=2),
        "Radius (Rsun)": sanitise_value(star_data.get("radius"), precision=2),
        "Luminosity (Lsun)": sanitise_value(star_data.get("luminosity"), precision=2),
        "Age (Gyr)": sanitise_value(star_data.get("age"), precision=2),
    }

    evo_hints = {
        "Stage Estimate": determine_stage(star_data.get("Teff"), star_data.get("logg")),
        "Rotation (km/s)": sanitise_value(star_data.get("vsini"), precision=1),
        "Metallicity Source": sanitise_value(star_data.get("metallicity_source")),
        "Surface Composition Note": sanitise_value(star_data.get("composition_note")),
    }

    observational = {}
    for key in ["obsid", "spectral_type", "SpT", "SpType", "phot_g_mean_mag", "distance"]:
        value = star_data.get(key)
        value = sanitise_value(value, precision=2)
        if value is not None:
            observational[key] = value

    payload = ""
    payload += format_prompt_block("Core Stellar Parameters", stellar_params)
    payload += format_prompt_block("Evolutionary Clues", evo_hints)
    payload += format_prompt_block("Observational Metadata", observational)
    return payload


def create_advanced_prompt(star_data: Dict[str, Any], focus: str, feedback: str = "") -> str:
    """Generate the Gemini prompt for advanced single-star questions."""
    prompt = """
You are an astrophysicist preparing rigorous question-answer pairs about stellar evolution.
The dataset must be extremely diverse and every statement must be scientifically correct.
Always double-check that the answer is consistent with the supplied data.
Create exactly one question and one answer about the star with the following requirements:
1. Write the question as two sentences explicitly labelled "Part 1:" and "Part 2:". Part 1 must ask the reader to predict the star's Teff, log g, and [Fe/H] without stating any numeric values. Part 2 must pose an open ended query about its evolutionary future: {focus}.
2. Do not copy any actual numeric values into the question text. Use qualitative phrasing such as "estimate the effective temperature" or "characterise the surface gravity" rather than quoting numbers.
3. Write the answer as two sentences labelled "Part 1:" and "Part 2:". Part 1 must report the predicted Teff, log g, and [Fe/H] using concise notation such as 'Teff ~ 6150 K'. Part 2 must provide the requested evolutionary interpretation.
4. Both the question and the answer must be at most {word_limit} words each. Use Teff/log g/[Fe/H] abbreviations to keep them concise.
5. Vary the wording and chosen evolutionary insights from example to example to keep the dataset diverse.
6. If data are uncertain, provide the most plausible astrophysical inference consistent with the numbers rather than fabricating new ones.

Return only a JSON object with the following schema:
{{
  "Question": "<question text>",
  "Answer": "<answer text>"
}}

Do not add markdown code fences or extra commentary. Count the words yourself before returning the JSON.
"""

    formatted_prompt = prompt.format(focus=focus, word_limit=WORD_LIMIT)
    if feedback:
        formatted_prompt += f"\nPrevious output issues to fix: {feedback.strip()}\n"

    payload = build_star_prompt_payload(star_data)
    return formatted_prompt + payload


def validate_response(data: Dict[str, Any]) -> Optional[str]:
    """Return an error message when validation fails."""
    if not isinstance(data, dict):
        return "Response was not a JSON object."

    question = data.get("Question")
    answer = data.get("Answer")

    if not question or not isinstance(question, str):
        return "Missing or invalid 'Question' field."
    if not answer or not isinstance(answer, str):
        return "Missing or invalid 'Answer' field."

    if "Part 1:" not in question or "Part 2:" not in question:
        return "Question must contain 'Part 1:' and 'Part 2:' segments."

    if "Part 1:" not in answer or "Part 2:" not in answer:
        return "Answer must contain 'Part 1:' and 'Part 2:' segments."

    q_words = count_words(question)
    a_words = count_words(answer)

    if q_words > WORD_LIMIT or a_words > WORD_LIMIT:
        return (f"Word limit exceeded (Question: {q_words} words, Answer: {a_words} words). "
                f"Both must be ≤ {WORD_LIMIT}.")

    return None


def convert_to_native(obj: Any) -> Any:
    """Convert numpy dtypes to native Python types so the output serialises cleanly."""
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
    """
    Prepare stellar data for prompting.
    Teff values are stored normalised to solar and need to be expressed in Kelvin.
    """
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


def generate_focus(rng: random.Random) -> str:
    """Select an evolutionary focus to encourage question diversity."""
    return rng.choice(EVOLUTIONARY_FOCI)


def generate_single_entry(star_data: Dict[str, Any],
                          idx: Any,
                          seed: int,
                          max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """Ask Gemini to craft an advanced QA pair for a single star."""
    client = _ensure_client()
    working_data = dict(star_data)

    try:
        teff = float(working_data.get("Teff"))
        logg = float(working_data.get("logg"))
        working_data["stage_flag"] = "giant" if not giant_cond(teff, logg) else "dwarf"
    except (TypeError, ValueError):
        working_data["stage_flag"] = None

    rng = make_rng(seed, idx)
    feedback = ""

    for attempt in range(1, max_retries + 1):
        focus = generate_focus(rng)
        prompt = create_advanced_prompt(working_data, focus, feedback)

        try:
            response = client.models.generate_content(
                contents=[prompt],
                model=MODEL_NAME,
                config={"response_mime_type": "application/json"}
            )
            response_text = getattr(response, "text", None)
            if not response_text:
                feedback = "Empty response text. Provide the JSON object."
                time.sleep(BASE_DELAY * attempt)
                continue

            parsed = parse_gemini_json_response(response_text)
            error = validate_response(parsed)
            if error:
                feedback = error
                time.sleep(BASE_DELAY * attempt)
                continue

            index_value: Any = int(idx) if isinstance(idx, (int, np.integer)) else idx
            return {
                "index": index_value,
                "question": parsed["Question"].strip(),
                "answer": parsed["Answer"].strip(),
                "word_counts": {
                    "question": count_words(parsed["Question"]),
                    "answer": count_words(parsed["Answer"]),
                },
                "stellar_data": convert_to_native(working_data),
            }

        except Exception as exc:
            feedback = f"Exception encountered: {exc}"
            time.sleep(BASE_DELAY * attempt)

    print(f"Failed to generate QA pair for index {idx}: {feedback}")
    return None


def _single_star_worker(args: Tuple[Any, Dict[str, Any], int, int]) -> Optional[Dict[str, Any]]:
    idx, star_dict, seed, max_retries = args
    return generate_single_entry(star_dict, idx, seed, max_retries)


def _worker_initializer() -> None:
    """Reset lazy-loaded resources for each worker process."""
    global _CLIENT
    _CLIENT = None


def generate_dataset(input_csv: str,
                     output_json: str,
                     limit: Optional[int],
                     seed: int,
                     max_retries: int,
                     processes: int) -> None:
    """Generate the advanced single-star QA dataset."""
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} stars from {input_csv}")

    if processes < 1:
        raise ValueError("Number of processes must be at least 1.")

    tasks: List[Tuple[Any, Dict[str, Any], int, int]] = []
    for idx, row in df.iterrows():
        star_dict = preprocess_star_dict(convert_to_native(row.to_dict()))
        tasks.append((idx, star_dict, seed, max_retries))

    results: List[Dict[str, Any]] = []

    if processes > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes, initializer=_worker_initializer) as pool:
            iterator = pool.imap_unordered(_single_star_worker, tasks)
            for entry in tqdm(iterator, total=len(tasks), desc="Generating advanced QA"):
                if entry:
                    results.append(entry)
    else:
        _worker_initializer()
        for task in tqdm(tasks, total=len(tasks), desc="Generating advanced QA"):
            entry = _single_star_worker(task)
            if entry:
                results.append(entry)

    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} entries to {output_json}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate advanced single-star evolution QA pairs using Gemini."
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
        help="Destination JSON file for the generated QA dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of rows to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling diversity hints.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=MAX_RETRIES,
        help="Maximum number of attempts per star.",
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
