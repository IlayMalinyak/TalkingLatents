#!/usr/bin/env python3
"""
Utility functions for generating follow-up questions and templated answers
that stay consistent with stellar parameter metadata.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

PHYSICAL_BOUNDS = {
    'Teff': (3000.0, 7500.0),
    'logg': (0.0, 5.0),
    'FeH': (-3.0, 0.5),
}

PARAM_SYNONYMS = {
    'Teff': ['temperature', 'effective temperature', 'Teff', 'stellar temperature', 'thermal profile'],
    'logg': ['surface gravity', 'logg', 'log g', 'gravity at the photosphere'],
    'FeH': ['metallicity', 'iron abundance', '[Fe/H]', 'FeH content'],
}

PARAM_DIRECTION_PHRASES = {
    'Teff': ['a hotter', 'a cooler'],
    'logg': ['higher', 'lower'],
    'FeH': ['more metal-rich', 'more metal-poor'],
}

PARAM_KEY_ALIASES = {
    'Teff': ['Teff', 'teff', 'teff_k', 'effective_temperature'],
    'logg': ['logg', 'log_g', 'log_g_surface'],
    'FeH': ['FeH', 'feh', '[Fe/H]', 'metallicity'],
}


def _format_param_value(value: Optional[float], param: str) -> Optional[str]:
    if value is None or math.isnan(value):
        return None
    if param == 'Teff':
        return f"{value:.0f} K"
    return f"{value:.2f}"


def _infer_stellar_type(params: Dict[str, Optional[float]]) -> str:
    """Infer stellar type from Teff and logg."""
    teff = params.get('Teff')
    logg = params.get('logg')

    if teff is None or math.isnan(teff):
        return "Unknown type"
    if logg is None or math.isnan(logg):
        logg = 4.0  # Assume main sequence if unknown

    # Determine spectral class from Teff
    if teff >= 7500:
        spectral = 'A'
    elif teff >= 6000:
        spectral = 'F'
    elif teff >= 5200:
        spectral = 'G'
    elif teff >= 3700:
        spectral = 'K'
    else:
        spectral = 'M'

    # Determine luminosity class from logg
    if logg < 1.0:
        lum_class = 'I'
        lum_name = 'supergiant'
    elif logg < 3.0:
        lum_class = 'III'
        lum_name = 'giant'
    elif logg < 4.0:
        lum_class = 'IV'
        lum_name = 'subgiant'
    else:
        lum_class = 'V'
        lum_name = 'dwarf'

    return f"{spectral}-type {lum_name}"


def _describe_star(params: Dict[str, Optional[float]]) -> str:
    fragments: List[str] = []
    for param in ['Teff', 'logg', 'FeH']:
        value = _format_param_value(params.get(param), param)
        if value is None:
            continue
        label = random.choice(PARAM_SYNONYMS[param])
        fragments.append(f"{label} around {value}")
    if not fragments:
        return "This star exhibits well-behaved stellar parameters typical of a stable photosphere."
    joined = ", ".join(fragments[:-1])
    if fragments[:-1]:
        joined = f"{joined}, and {fragments[-1]}"
    else:
        joined = fragments[-1]
    return f"This star shows {joined}, consistent with a balanced stellar atmosphere."


def _clamp(value: float, param: str) -> float:
    low, high = PHYSICAL_BOUNDS.get(param, (value, value))
    return max(low, min(high, value))


def _build_star_type_spec(params: Dict[str, Optional[float]],
                          rng: random.Random,
                          include_answer: bool) -> Optional[Dict[str, str]]:
    if not params:
        return None

    # Simple questions without parameter values
    question_templates = [
        "What stellar classification best describes this object?",
        "What stellar type best describes this object?",
        "What is the stellar classification of this object?",
        "What is the stellar type of this object?",
    ]
    question = rng.choice(question_templates)

    spec = {'type': 'star_type', 'question': question}
    if include_answer:
        # Short answer: just the stellar type
        spec['answer'] = _infer_stellar_type(params)
    return spec


def _build_contrast_spec(params: Dict[str, Optional[float]],
                         rng: random.Random,
                         include_answer: bool) -> Optional[Dict[str, str]]:
    available = [p for p, v in params.items() if v is not None and not math.isnan(v)]
    if len(available) < 1:
        return None
    keep_params = rng.sample(available, k=min(2, len(available)))
    change_candidates = [p for p in available if p not in keep_params] or available
    change_param = rng.choice(change_candidates)
    direction_words = PARAM_DIRECTION_PHRASES.get(change_param, ['higher', 'lower'])
    direction = rng.choice(direction_words)
    descriptors = []
    for param in keep_params:
        formatted = _format_param_value(params.get(param), param)
        if formatted:
            label = rng.choice(PARAM_SYNONYMS[param])
            descriptors.append(f"{label} near {formatted}")
    similarity_clause = " and ".join(descriptors) if descriptors else "very similar baseline properties"
    question = (
        f"Describe a star that would maintain {similarity_clause} but exhibit {direction} "
        f"{rng.choice(PARAM_SYNONYMS.get(change_param, [change_param]))} compared to this target."
    )
    spec = {'type': 'contrastive', 'question': question, 'change_param': change_param, 'direction': direction}
    if include_answer:
        original = params.get(change_param)
        description = _describe_star(params)
        if original is not None and not math.isnan(original):
            low, high = PHYSICAL_BOUNDS.get(change_param, (original, original))
            delta = 0.15 * (high - low)
            new_value = _clamp(original + (delta if 'higher' in direction or 'hotter' in direction else -delta), change_param)
            change_text = _format_param_value(new_value, change_param)
            if change_text:
                description = (
                    f"{description} Adjust the {change_param} to about {change_text} to satisfy the follow-up request."
                )
        spec['answer'] = description
    return spec


def create_follow_up_specs(params: Dict[str, Optional[float]],
                           rng: random.Random,
                           max_pairs: int = 2,
                           include_answers: bool = False) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    star_spec = _build_star_type_spec(params, rng, include_answers)
    if star_spec is not None:
        specs.append(star_spec)
    if len(specs) < max_pairs:
        contrast_spec = _build_contrast_spec(params, rng, include_answers)
        if contrast_spec is not None:
            specs.append(contrast_spec)
    return specs


__all__ = [
    'PHYSICAL_BOUNDS',
    'PARAM_SYNONYMS',
    'PARAM_DIRECTION_PHRASES',
    'PARAM_KEY_ALIASES',
    'create_follow_up_specs',
]
