#!/usr/bin/env python3
"""
Local runner for the infer_simple_questions_multitok_ablation workload.

This mirrors the logic in cluster/infer_simple_questions_multitok_ablation.sbatch
so the sweep can be exercised without queueing a Slurm job.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


EXPERIMENTS: List[str] = [
    "abl_single_lambda1",
    "abl_single_lambda0p5",
    # "abl_combined_lambda1",
    # "abl_combined_lambda0p5",
    "abl_advanced_lambda1",
    "abl_advanced_lambda0p5",
]


def _collect_checkpoint_candidates(base_dir: Path, exp_name: str) -> List[Path]:
    """Return sorted candidate checkpoint paths, mimicking `find -maxdepth 2`."""
    patterns = [
        exp_name + ".pth",
        f"*/{exp_name}.pth",
        f"*/*/{exp_name}.pth",
    ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(base_dir.glob(pattern))
    try:
        return sorted(matches, key=lambda path: path.stat().st_mtime)
    except OSError:
        return sorted(matches)


def find_checkpoint(base_log_dir: Path, exp_name: str) -> Optional[Path]:
    if not base_log_dir.is_dir():
        return None
    candidates = _collect_checkpoint_candidates(base_log_dir, exp_name)
    if not candidates:
        return None

    exp_lower = exp_name.lower()
    if "combined" in exp_lower:
        for candidate in reversed(candidates):
            parts_lower = [part.lower() for part in candidate.parts]
            if any("combined" in part for part in parts_lower):
                return candidate
    return candidates[-1]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    base_log_dir = Path(
        os.environ.get("BASE_LOG_DIR", "/data/TalkingLatents/logs")
    ).expanduser()
    llm_root = os.environ.get("LLM_ROOT", "/data/.llama")
    inference_script = repo_root / "src" / "inference.py"

    print(f"Running inference for experiments: {', '.join(EXPERIMENTS)}")

    for exp_name in EXPERIMENTS:
        print(f"=== Processing {exp_name} ===")
        checkpoint_path = find_checkpoint(base_log_dir, exp_name)
        if checkpoint_path is None:
            print(
                f"Warning: checkpoint for {exp_name} not found under {base_log_dir}; skipping."
            )
            continue

        checkpoint_dir = checkpoint_path.parent
        output_dir = checkpoint_dir / f"inference_{exp_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output dir: {output_dir}")

        cmd = [
            sys.executable,
            "-u",
            str(inference_script),
            "--checkpoint_path",
            str(checkpoint_path),
            "--output_dir",
            str(output_dir),
            "--interpolation_num_pairs",
            "20",
            "--interpolation_min_teff_diff",
            "1000",
            "--llm_root",
            llm_root,
        ]

        subprocess.run(cmd, check=True)

    print("Inference sweep complete.")


if __name__ == "__main__":
    main()
