import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import json
import os
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from functools import partial
from google import genai
from itertools import chain

warnings.filterwarnings('ignore')

# Import your existing functions (assuming they're available)
from data.stellar_evolution import generate_evolutionary_tracks, tracks_to_dataframe


def merge_dataframes_and_filter_array(df1, df2, right_on, left_on, array):
    """
    Merge two dataframes on all columns and filter corresponding numpy array
    Parameters:
    - df1: First dataframe (berger_lamost)
    - df2: Second dataframe (features_meta)
    - array: Numpy array aligned with df2 (features)
    Returns:
    - merged_df: Merged dataframe
    - filtered_array: Filtered numpy array
    """
    # Track original indices
    df2_indexed = df2.reset_index().rename(columns={'index': 'original_idx'})
    # Merge on all original columns from df2
    merged = pd.merge(df1, df2_indexed, right_on=right_on, left_on=left_on, how='inner', suffixes=['', '_2'])
    # Filter array based on surviving indices
    surviving_indices = merged['original_idx'].values
    filtered_array = array[surviving_indices]
    # Clean up merged dataframe
    final_df = merged.drop('original_idx', axis=1).reset_index(drop=True)
    return final_df, filtered_array


def giant_cond(teff, logg):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :return: boolean
    """
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh


# Your existing Gemini setup
def init_gemini_client():
    """Initialize Gemini client using your existing setup"""
    try:
        with open('google_api.txt', 'r') as f:
            google_api_key = f.read().strip()
        os.environ["GOOGLE_API_KEY"] = google_api_key
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    except FileNotFoundError:
        print("Warning: google_api.txt not found. Gemini client not initialized.")
        return None


# Initialize client globally as in your code
try:
    with open('google_api.txt', 'r') as f:
        google_api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = google_api_key
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
except:
    client = None
    print("Warning: Could not initialize Gemini client")


def create_star_prompt(example):
    """
    Your existing function adapted for stellar evolution data.
    Convert stellar metadata to a comprehensive structured prompt for an LLM.

    Args:
        example: Dictionary containing stellar metadata

    Returns:
        str: Formatted prompt with all relevant stellar information
    """
    # Base prompt
    base_prompt = """
Describe this star in detail. Act like a scientist. Do not converse please, just give your description in a scientific tone.

Here is additional information about this star:
"""

    # Helper function to format values appropriately
    def format_value(key, value):
        # Skip None values, PIL image objects, and index values
        if value is None or str(type(value)).find('PIL') >= 0 or key == '__index_level_0__':
            return None

        # Handle numeric values
        if isinstance(value, (int, float)):
            # Format temperatures
            if key in ['Teff', 'temperature']:
                return f"{value:.0f}"
            # Format surface gravity
            elif key in ['logg', 'surface_gravity']:
                return f"{value:.2f}"
            # Format metallicity
            elif key in ['feh', 'metallicity', 'FeH']:
                return f"{value:.2f}"
            # Format masses
            elif key in ['mass', 'Mstar', 'initial_mass']:
                return f"{value:.2f}"
            # Format luminosity
            elif key in ['Lstar', 'luminosity', 'L']:
                return f"{value:.3f}"
            # Format age
            elif key in ['age']:
                return f"{value:.2f}"
            # Format coordinates
            elif key in ['_RA', '_DE']:
                return f"{value:.6f}"
            # Format error values
            elif key.find('err') > 0 or key.startswith('e_') or key.startswith('sig'):
                return f"{value:.6f}"
            # Default number formatting
            return f"{value:.4f}" if isinstance(value, float) else f"{value}"

        # String values
        return value

    # Group metadata into categories
    categories = {
        "Basic Information": [
            'KID', 'star_id', '_RA', '_DE', 'track_id'
        ],
        "Stellar Parameters": [
            'Teff', 'logg', 'FeH', 'feh', 'metallicity'
        ],
        "Physical Properties": [
            'Mstar', 'mass', 'initial_mass', 'Lstar', 'luminosity', 'L'
        ],
        "Evolution Properties": [
            'age', 'Main-Sequence', 'Giant', 'main_sequence', 'giant', 'evolutionary_phase'
        ],
    }

    # Build the prompt with organized scientific data
    formatted_prompt = base_prompt

    for category, keys in categories.items():
        # Check if we have any values for this category
        category_values = {}
        for key in keys:
            if key in example and example[key] is not None:
                formatted_value = format_value(key, example[key])
                if formatted_value is not None:
                    category_values[key] = formatted_value

        # If we have values, add the category
        if category_values:
            formatted_prompt += f"\n\n{category}:"
            for key, value in category_values.items():
                # Format key for better readability
                display_key = key.replace('_', ' ').replace('-', ' ')
                display_key = ' '.join(word.capitalize() for word in display_key.split())
                formatted_prompt += f"\n- {display_key}: {value}"

    # Add a final instruction
    formatted_prompt += """

Based on this information and the STELLAR_DATA, provide a detailed scientific description of this star. Do not include references to the values above, simply use them to roughly guide your description.
Include the word STAR_DATA_0 to identify the star. It will use as a placeholder for numerical data about the star.
Your response must be valid JSON following this structure:

{
"Question": "Question about the star...",
"Answer": "Detailed scientific description of the star..."
}


"""

    return formatted_prompt


def create_evolutionary_star_prompt(example):
    """
    Create a prompt for a star with multiple evolutionary stages.

    Args:
        example: Dictionary containing stellar metadata with multiple stages

    Returns:
        str: Formatted prompt describing stellar evolution across stages
    """
    # Base prompt for evolutionary description
    base_prompt = """
Describe the evolutionary path of this star across multiple stages. Act like a scientist providing a detailed analysis of stellar evolution.

Here is information about this star at different evolutionary stages:
"""

    # Helper function to format values appropriately
    def format_value(key, value):
        if value is None:
            return None

        if isinstance(value, (int, float)):
            if key in ['Teff', 'temperature']:
                return f"{value:.0f} K"
            elif key in ['logg', 'surface_gravity']:
                return f"{value:.2f} dex"
            elif key in ['feh', 'metallicity', 'FeH']:
                return f"{value:.2f}"
            elif key in ['mass', 'Mstar', 'initial_mass']:
                return f"{value:.2f} M☉"
            elif key in ['Lstar', 'luminosity', 'L']:
                return f"{value:.3f} L☉"
            elif key in ['age']:
                return f"{value:.2f} Gyr"
            return f"{value:.4f}" if isinstance(value, float) else f"{value}"
        return str(value)

    # Get evolutionary stages
    stages = example['evolutionary_stages']
    initial_mass = example['initial_mass']
    metallicity = example['metallicity']

    formatted_prompt = base_prompt

    # Add initial star properties
    formatted_prompt += f"""

Initial Stellar Properties:
- Initial Mass: {initial_mass:.2f} M☉
- Metallicity [Fe/H]: {metallicity:.2f}
"""

    # Add each evolutionary stage
    formatted_prompt += f"\nEvolutionary Sequence ({len(stages)} stages observed):\n"

    for i, stage in enumerate(stages):
        age = stage['age']
        teff = stage['Teff']
        logg = stage['logg']
        luminosity = stage['L']
        phase = stage['evolutionary_phase']

        formatted_prompt += f"""
Stage {i + 1} - Age {age:.2f} Gyr ({phase}):
- Effective Temperature: {teff:.0f} K
- Surface Gravity: {logg:.2f} dex
- Luminosity: {luminosity:.3f} L☉
"""

    # Add evolutionary context
    age_span = stages[-1]['age'] - stages[0]['age']
    temp_change = stages[-1]['Teff'] - stages[0]['Teff']
    lum_change = stages[-1]['L'] - stages[0]['L']

    formatted_prompt += f"""
Evolutionary Summary:
- Time span observed: {age_span:.2f} Gyr
- Temperature evolution: {temp_change:+.0f} K
- Luminosity evolution: {lum_change:+.3f} L☉
"""

    # Add final instruction
    formatted_prompt += """

Based on this evolutionary sequence, provide a detailed scientific description that covers:
1. The star's evolutionary path and current stages observed
2. Physical changes occurring between stages
3. Underlying stellar physics driving these changes
4. Predictions for future evolutionary stages
5. How mass and metallicity influence this evolutionary path
6. Include the word STAR_DATA_S as a placeholder for numerical data about the star. S is an indicator for the different evolutionary stages (each stage has different STAR_DATA). for example STAR_DATA_0 refers to the data of the star at the first evolutionary stage and STAR_DATA_1 refers to the data of the star at the second evolutionary stage. Use it also to identify the star at the beginning of the question (don't use the track id). 

Focus on the evolutionary progression and the physical processes causing the observed changes in stellar properties.
Your response must be valid JSON following this structure:
{
"Question": "Question about the star...",
"Answer": "Detailed scientific description of the star..."
}

"""

    return formatted_prompt


def create_stellar_conversation_prompt(example):
    """
    Your existing function adapted for stellar evolution.
    Generate a prompt for creating a synthetic conversation about a star
    between a curious human and an astronomical AI assistant.

    Args:
        example: Dictionary containing stellar metadata

    Returns:
        str: Formatted prompt to generate an entertaining educational conversation
    """
    # First get all the organized stellar data using our existing function
    star_data = create_star_prompt(example).split(
        "Here is additional information about this star:\n\n")[1].split(
        "\n\nBased on this information")[0]

    conversation_prompt = f"""
Generate an entertaining and educational conversation between a curious human and an astronomical AI assistant passionate about stars and stellar evolution. The conversation should use the following stellar data as a rough guide (but not reference it directly):
{star_data}

The conversation should follow this pattern:

1. Human asks about the star and its evolutionary stage
2. AI explains the basic features enthusiastically, using a friendly tone with occasional astronomy puns
3. Human asks a follow-up question about something specific (evolutionary phase, stellar mass, temperature, etc.)
4. AI provides a more detailed explanation, connecting the feature to broader stellar evolution concepts
5. Human expresses amazement and asks another question
6. AI shares an interesting fact or comparison about this type of star or evolutionary stage

Guidelines for AI's personality:
- Enthusiastic and passionate about stellar astronomy and evolution
- Uses accessible language but includes proper scientific terminology
- Occasionally uses space-related puns or expressions ("out of this world," "stellar example," etc.)
- Anthropomorphizes stars occasionally ("this star seems to be having a mid-life crisis")
- Connects observations to broader astronomical concepts
- Expresses wonder at stellar evolution and cosmic beauty

Both Dr. Stella and the student must include the word STAR_DATA everytime they refer to the data of the star
(for example "I'm curious about this star: STAR_DATA"). This will use
as a placeholder for numerical data about the star

Your response must be valid JSON following this structure:
{{
  "conversation": [
    {{
      "speaker": "human",
      "text": "Question about the star..."
    }},
    {{
      "speaker": "AI",
      "text": "Enthusiastic, educational response about the star..."
    }},
    // Additional turns following the pattern above
  ]
}}

The conversation should be informative while remaining scientifically accurate about stellar evolution. 
Do not include references to the values above, simply use them to roughly guide your conversation.
"""
    return conversation_prompt


def create_evolutionary_conversation_prompt(example):
    """
    Generate a conversation prompt for stellar evolution across multiple stages.

    Args:
        example: Dictionary containing stellar metadata with multiple stages

    Returns:
        str: Formatted prompt for evolutionary conversation
    """
    stages = example['evolutionary_stages']
    initial_mass = example['initial_mass']
    metallicity = example['metallicity']

    # Create summary of evolutionary data
    stage_summary = f"Initial mass: {initial_mass:.2f} M☉, Metallicity: {metallicity:.2f}\n"
    for i, stage in enumerate(stages):
        stage_summary += f"Age {stage['age']:.2f} Gyr: {stage['Teff']:.0f} K, log g = {stage['logg']:.2f}, L = {stage['L']:.3f} L☉ ({stage['evolutionary_phase']})\n"

    conversation_prompt = f"""
Generate an educational conversation between a curious astronomy student and Dr. Stella, an expert in stellar evolution. The conversation should be based on observations of a star at multiple evolutionary stages:

Stellar Evolution Data:
{stage_summary}

The conversation should follow this pattern:
1. Student asks about this star's evolutionary journey
2. Dr. Stella explains the overall evolutionary path enthusiastically
3. Student asks about specific changes between stages
4. Dr. Stella explains the physical processes driving the evolution
5. Student inquires about predictions for future evolution
6. Dr. Stella discusses the role of mass and metallicity in stellar evolution

Guidelines for Dr. Stella:
- Enthusiastic about stellar evolution and physics
- Uses accessible language with proper scientific terminology
- Explains the "why" behind evolutionary changes
- Occasionally uses analogies ("like a cosmic aging process")
- Connects observations to broader astrophysical principles
- Emphasizes the time scales and physical processes involved

Both Dr. Stella and the student must include the word STAR_DATA everytime they refer to the data of the star
(for example "I'm curious about this star: STAR_DATA"). This will use
as a placeholder for numerical data about the star

Your response must be valid JSON:
{{
  "conversation": [
    {{
      "speaker": "student",
      "text": "Question about stellar evolution..."
    }},
    {{
      "speaker": "dr_stella", 
      "text": "Educational response about stellar evolution..."
    }}
  ]
}}

Focus on the evolutionary progression and physical understanding rather than just listing parameters.
"""
    return conversation_prompt


def caption_star(features, information, conversation=False):
    """
    Enhanced caption function that handles both single-stage and evolutionary prompts
    """
    global client
    if client is None:
        print("Warning: Gemini client not available")
        return None

    # Determine prompt type and create appropriate prompt
    prompt_type = information.get('prompt_type', 'single_stage')

    if prompt_type == 'evolutionary':
        # Use evolutionary prompts
        prompt = create_evolutionary_conversation_prompt(
            information) if conversation else create_evolutionary_star_prompt(information)
        # Handle multiple features for evolutionary examples
        if isinstance(features, np.ndarray) and len(features.shape) > 1:
            features_text = f"Spectral features for {len(features)} evolutionary stages"
        else:
            features_text = f"Spectral features: {features[:10].tolist() if hasattr(features, 'tolist') else features}"
    else:
        # Use single-stage prompts
        prompt = create_stellar_conversation_prompt(information) if conversation else create_star_prompt(information)
        # Convert features to text representation for Gemini
        if isinstance(features, np.ndarray):
            features_text = f"Spectral features (first 10): {features[:10].tolist()}"
        else:
            features_text = f"Features: {features}"

    # Generate response from Gemini
    try:
        response = client.models.generate_content(
            contents=[prompt, features_text],
            model="gemini-2.0-flash",
        )
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

    if conversation:
        for attempt in range(10):
            if attempt > 0:
                print(f"Attempting try {attempt}/10 for stellar conversation")
            try:
                # Get response text
                text = response.text

                # Remove markdown code blocks if present
                if text.startswith("```json") and text.endswith("```"):
                    text = text[7:-3]  # Remove ```json at start and ``` at end
                elif text.startswith("```") and text.endswith("```"):
                    text = text[3:-3]  # Remove ``` at start and end

                # Trim whitespace
                text = text.strip()

                # Parse JSON response
                conversation_data = json.loads(text)
                return conversation_data
            except json.JSONDecodeError as e:
                # Fallback if JSON parsing fails
                print(f"Retrying due to failure to parse JSON response: {e}")

                try:
                    response = client.models.generate_content(
                        contents=[prompt, features_text],
                        model="gemini-2.0-flash",
                    )
                except Exception as e:
                    print(f"Error on retry: {e}")
                    return None
        return None
    else:
        try:
            return response.text
        except Exception as e:
            print(f"Error getting response text: {e}")
            return None


def process_example(example_data, conversation=False):
    """
    Enhanced processing function that handles both single-stage and evolutionary examples
    """
    try:
        features = example_data['features']
        information = example_data['information']
        prompt_type = information.get('prompt_type', 'single_stage')

        result = caption_star(features, information, conversation)

        # Create return structure
        return_data = {
            'track_id': information.get('track_id', 'unknown'),
            'star_id': information.get('star_id', 'unknown'),
            'prompt_type': prompt_type,
            'result': result,
            'information': information
        }

        # Handle features differently for evolutionary vs single-stage
        if prompt_type == 'evolutionary':
            return_data['features'] = features.tolist() if isinstance(features, np.ndarray) else features
            return_data['n_stages'] = information.get('n_stages', 1)
            return_data['stage_mapping'] = information.get('stage_mapping', [])
        else:
            return_data['features'] = features.tolist() if isinstance(features, np.ndarray) else features

        return return_data

    except Exception as e:
        print(f"Error processing example: {e}")
        return None


def save_results(results, filename):
    """
    Your existing function for saving captioning results to JSON file
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


class EvolutionaryDatasetCreator:
    """
    Creates a dataset matching evolutionary tracks with real stellar observations.
    Integrates with your existing prompt generation pipeline.
    """

    def __init__(self, real_stars_df, real_features,
                 feature_columns=['Teff', 'logg', 'L', 'Mstar'],
                 weights=None):
        """
        Initialize the dataset creator.

        Parameters:
        -----------
        real_stars_df : pd.DataFrame
            DataFrame with real star observations containing stellar parameters
        real_features : np.array
            Features array of shape (n_stars, hidden_dim) corresponding to real_stars_df
        feature_columns : list
            Column names in real_stars_df to use for matching
        weights : dict or None
            Weights for each parameter in matching (default: equal weights)
        """
        self.real_stars_df = real_stars_df.copy()
        self.real_features = real_features
        self.feature_columns = feature_columns

        # Set default weights if none provided
        if weights is None:
            self.weights = {col: 1.0 for col in feature_columns}
        else:
            self.weights = weights

        # Prepare real star data for matching
        self._prepare_real_star_data()

    def _prepare_real_star_data(self):
        """Prepare and normalize real star data for efficient matching."""
        print("Preparing real star data for matching...")

        # Extract relevant columns and handle missing values
        self.real_params = self.real_stars_df[self.feature_columns].copy()

        # Remove rows with any NaN values in the matching columns
        valid_mask = ~self.real_params.isnull().any(axis=1)
        self.real_params = self.real_params[valid_mask]
        self.real_features_clean = self.real_features[valid_mask]
        self.real_stars_clean = self.real_stars_df[valid_mask].reset_index(drop=True)

        print(f"Found {len(self.real_params)} valid real stars for matching")

        # Normalize parameters for fair comparison
        self.scaler = StandardScaler()
        self.real_params_normalized = self.scaler.fit_transform(self.real_params)

        # Apply weights
        weight_array = np.array([self.weights[col] for col in self.feature_columns])
        self.real_params_weighted = self.real_params_normalized * weight_array

    def find_closest_real_star(self, synthetic_params, max_deviations=None, min_deviations=None):
        """
        Find the closest real star to synthetic parameters using individual parameter thresholds.

        Parameters:
        -----------
        synthetic_params : dict
            Dictionary with stellar parameters {parameter: value}
        max_deviations : dict or None
            Maximum allowed absolute deviation for each parameter
            Example: {'Teff': 200, 'logg': 0.3, 'L': 0.5, 'Mstar': 0.2}
            If None, uses default values
        min_deviations : dict or None
            Minimum required absolute deviation for each parameter to avoid exact duplicates
            If None, uses small default values

        Returns:
        --------
        dict or None: Information about the matched real star, or None if no valid match
        """
        # Set default maximum deviations if not provided
        if max_deviations is None:
            max_deviations = {
                'Teff': 300,  # ±300 K
                'logg': 0.4,  # ±0.4 dex
                'L': 0.8,  # ±0.8 solar luminosities
                'Mstar': 0.3  # ±0.3 solar masses
            }

        # Set default minimum deviations if not provided
        if min_deviations is None:
            min_deviations = {
                'Teff': 10,  # At least 10 K difference
                'logg': 0.01,  # At least 0.01 dex difference
                'L': 0.01,  # At least 0.01 L_sun difference
                'Mstar': 0.01  # At least 0.01 M_sun difference
            }

        # Extract synthetic parameter values
        synth_values = np.array([synthetic_params[col] for col in self.feature_columns])

        # Create masks for each parameter constraint
        valid_mask = np.ones(len(self.real_params), dtype=bool)

        for i, param in enumerate(self.feature_columns):
            real_values = self.real_params.iloc[:, i].values
            synth_value = synth_values[i]

            # Calculate absolute differences for this parameter
            abs_diff = np.abs(real_values - synth_value)

            # Apply maximum deviation constraint
            max_dev = max_deviations.get(param, np.inf)
            max_constraint = abs_diff <= max_dev

            # Apply minimum deviation constraint
            min_dev = min_deviations.get(param, 0)
            min_constraint = abs_diff >= min_dev

            # Combine constraints for this parameter
            param_valid = max_constraint & min_constraint
            valid_mask = valid_mask & param_valid

        # Check if any stars meet all constraints
        if not np.any(valid_mask):
            return None

        # Among valid stars, find the one with minimum weighted distance
        valid_indices = np.where(valid_mask)[0]
        valid_real_params = self.real_params.iloc[valid_indices].values

        # Calculate weighted distances only for valid stars
        synth_tiled = np.tile(synth_values, (len(valid_indices), 1))
        differences = valid_real_params - synth_tiled

        # Apply weights to differences
        weight_array = np.array([self.weights[col] for col in self.feature_columns])
        weighted_differences = differences * weight_array

        # Calculate weighted Euclidean distances
        distances = np.sqrt(np.sum(weighted_differences ** 2, axis=1))

        # Find the closest valid match
        closest_valid_idx = np.argmin(distances)
        closest_idx = valid_indices[closest_valid_idx]

        # Calculate parameter-wise differences for the chosen star
        param_differences = {}
        for i, param in enumerate(self.feature_columns):
            real_val = self.real_params.iloc[closest_idx, i]
            synth_val = synth_values[i]
            param_differences[param] = {
                'real_value': real_val,
                'synthetic_value': synth_val,
                'absolute_difference': abs(real_val - synth_val),
                'relative_difference': abs(real_val - synth_val) / synth_val if synth_val != 0 else np.inf
            }

        return {
            'real_star_idx': closest_idx,
            'weighted_distance': distances[closest_valid_idx],
            'parameter_differences': param_differences,
            'real_params': self.real_params.iloc[closest_idx].to_dict(),
            'real_features': self.real_features_clean[closest_idx],
            'real_star_data': self.real_stars_clean.iloc[closest_idx].to_dict(),
            'n_candidates': len(valid_indices)  # How many stars met the criteria
        }

    def create_matched_dataset(self, n_evolutionary_tracks=100,
                               age_points=10, track_params=None,
                               max_deviations=None, min_deviations=None,
                               min_matches_per_track=3, create_evolutionary_prompts=True,
                               min_stages_for_evolution=2):
        """
        Create dataset by matching evolutionary tracks with real stars.
        Now supports both single-stage and multi-stage evolutionary prompts.

        Parameters:
        -----------
        create_evolutionary_prompts : bool
            Whether to create evolutionary prompts for stars with multiple matches
        min_stages_for_evolution : int
            Minimum number of matched stages required to create evolutionary prompt

        Returns:
        --------
        list: List of examples ready for your process_example function
        """
        # Set default track parameters
        if track_params is None:
            track_params = {
                'n_stars': n_evolutionary_tracks,
                'age_points': age_points,
                'm_min': 0.5,
                'm_max': 2.0,
                'min_age': 0.8,
                'max_age': 10.0,
                'feh_range': (-0.5, 0.3),
                'grid_name': 'fastlaunch',
                'method': 'nearest'
            }

        print(f"Generating {n_evolutionary_tracks} evolutionary tracks...")

        # Generate evolutionary tracks
        tracks = generate_evolutionary_tracks(**track_params)
        tracks_df = tracks_to_dataframe(tracks)

        print(f"Generated tracks with {len(tracks_df)} total points")
        print(f"Searching for real star matches...")

        # Print matching criteria
        if max_deviations:
            print("Maximum allowed deviations:")
            for param, dev in max_deviations.items():
                print(f"  {param}: ±{dev}")

        # Store examples in your format
        single_stage_examples = []
        evolutionary_examples = []
        track_success_count = 0
        total_matches = 0

        # Process each evolutionary track
        unique_stars = tracks_df['star_id'].unique()

        for star_id in tqdm(unique_stars, desc="Processing evolutionary tracks"):
            star_track = tracks_df[tracks_df['star_id'] == star_id].copy()
            star_matches = []

            # Try to match each evolutionary stage of this star
            for stage_num, stage in star_track.iterrows():
                # Prepare parameters for matching
                synthetic_params = {
                    'Teff': stage['Teff'],
                    'logg': stage['logg'],
                    'L': stage['L'],  # Use your column naming
                    'Mstar': stage['mass']  # Use mass as Mstar estimate
                }

                # Find closest real star
                match = self.find_closest_real_star(
                    synthetic_params,
                    max_deviations=max_deviations,
                    min_deviations=min_deviations
                )

                if match is not None:
                    total_matches += 1

                    # Store match info with stage data
                    match_info = {
                        'stage_data': stage,
                        'match_result': match,
                        'age': stage['age'],
                        'Teff': stage['Teff'],
                        'logg': stage['logg'],
                        'L': stage['L'],
                        'mass': stage['mass'],
                        'feh': stage['feh'],
                        'evolutionary_phase': 'main_sequence' if giant_cond(stage['Teff'], stage['logg']) else 'giant'
                    }
                    star_matches.append(match_info)

            # Process this star's matches
            if len(star_matches) >= min_matches_per_track:
                track_success_count += 1

                # Sort matches by age
                star_matches.sort(key=lambda x: x['age'])

                # Create single-stage examples
                for match_info in star_matches:
                    stage = match_info['stage_data']
                    match = match_info['match_result']

                    information = {
                        'track_id': int(star_id),
                        'star_id': f"track_{star_id}_age_{stage['age']:.2f}",
                        'track_mass': stage['mass'],
                        'feh': stage['feh'],
                        'metallicity': stage['feh'],
                        'age': stage['age'],
                        'Teff': stage['Teff'],
                        'logg': stage['logg'],
                        'L': stage['L'],
                        'luminosity': stage['L'],
                        'Lstar': stage['L'],  # Your column naming
                        'Mstar': stage['mass'],
                        'mass': stage['mass'],
                        'initial_mass': stage['mass'],
                        'Main-Sequence': float(giant_cond(stage['Teff'], stage['logg'])),
                        'Giant': float(1 - (giant_cond(stage['Teff'], stage['logg']))),
                        'main_sequence': float(giant_cond(stage['Teff'], stage['logg'])),
                        'giant': float(1 - (giant_cond(stage['Teff'], stage['logg']))),
                        'evolutionary_phase': match_info['evolutionary_phase'],
                        'match_distance': match['weighted_distance'],
                        'parameter_differences': match['parameter_differences'],
                        'real_star_params': match['real_params'],
                        'n_candidates': match['n_candidates'],
                        'prompt_type': 'single_stage'
                    }

                    example = {
                        'features': match['real_features'],
                        'information': information
                    }
                    single_stage_examples.append(example)

                # Create evolutionary prompt if enough stages
                if create_evolutionary_prompts and len(star_matches) >= min_stages_for_evolution:
                    # Prepare evolutionary example
                    evolutionary_stages = []
                    all_features = []
                    stage_mapping = []

                    for i, match_info in enumerate(star_matches):
                        stage_info = {
                            'age': match_info['age'],
                            'Teff': match_info['Teff'],
                            'logg': match_info['logg'],
                            'L': match_info['L'],
                            'evolutionary_phase': match_info['evolutionary_phase']
                        }
                        evolutionary_stages.append(stage_info)
                        all_features.append(match_info['match_result']['real_features'])
                        stage_mapping.append({
                            'stage_index': i,
                            'age': match_info['age'],
                            'real_star_idx': int(match_info['match_result']['real_star_idx'])
                        })

                    # Create evolutionary information
                    evolutionary_info = {
                        'track_id': int(star_id),
                        'star_id': f"track_{star_id}_evolutionary",
                        'initial_mass': star_matches[0]['mass'],
                        'metallicity': star_matches[0]['feh'],
                        'evolutionary_stages': evolutionary_stages,
                        'n_stages': len(evolutionary_stages),
                        'age_span': star_matches[-1]['age'] - star_matches[0]['age'],
                        'stage_mapping': stage_mapping,
                        'prompt_type': 'evolutionary'
                    }

                    evolutionary_example = {
                        'features': np.array(all_features),  # Multiple features
                        'information': evolutionary_info
                    }
                    evolutionary_examples.append(evolutionary_example)

        print(f"Successfully matched {track_success_count}/{len(unique_stars)} tracks")
        print(f"Single-stage examples: {len(single_stage_examples)}")
        print(f"Evolutionary examples: {len(evolutionary_examples)}")
        print(f"Total successful matches: {total_matches}")

        # Combine examples
        all_examples = single_stage_examples + evolutionary_examples

        # Print matching statistics
        if single_stage_examples:
            print("\nSingle-stage matching quality statistics:")
            all_diffs = {}
            for param in self.feature_columns:
                diffs = [ex['information']['parameter_differences'][param]['absolute_difference']
                         for ex in single_stage_examples]
                all_diffs[param] = diffs
                print(f"  {param}: mean=±{np.mean(diffs):.3f}, max=±{np.max(diffs):.3f}")

        return all_examples


def main(real_stars_df, real_features, n_tracks=100, age_points=10,
         output_file='stellar_evolution_dataset.json', conversation=False,
         use_multiprocessing=True, n_processes=4, max_deviations=None, min_deviations=None,
         create_evolutionary_prompts=True, min_stages_for_evolution=2):
    """
    Main function using your existing pipeline structure with evolutionary prompt support.

    Parameters:
    -----------
    create_evolutionary_prompts : bool
        Whether to create multi-stage evolutionary prompts
    min_stages_for_evolution : int
        Minimum number of matched stages to create evolutionary prompt
    """

    # Create the dataset creator
    creator = EvolutionaryDatasetCreator(
        real_stars_df=real_stars_df,
        real_features=real_features,
        feature_columns=['Teff', 'logg', 'L', 'Mstar'],
        weights={'Teff': 1.0, 'logg': 1.5, 'L': 1.0, 'Mstar': 2.0}
    )

    # Create matched examples
    print("Creating matched evolutionary dataset...")
    examples = creator.create_matched_dataset(
        n_evolutionary_tracks=n_tracks,
        age_points=age_points,
        max_deviations=max_deviations,
        min_deviations=min_deviations,
        min_matches_per_track=3,
        create_evolutionary_prompts=create_evolutionary_prompts,
        min_stages_for_evolution=min_stages_for_evolution
    )

    if len(examples) == 0:
        print("No valid examples created. Try relaxing the matching criteria.")
        return

    # Separate single-stage and evolutionary examples for reporting
    single_stage = [ex for ex in examples if ex['information'].get('prompt_type') == 'single_stage']
    evolutionary = [ex for ex in examples if ex['information'].get('prompt_type') == 'evolutionary']

    print(f"Processing {len(examples)} examples with Gemini...")
    print(f"  - Single-stage examples: {len(single_stage)}")
    print(f"  - Evolutionary examples: {len(evolutionary)}")

    # Process examples using your existing pipeline
    if use_multiprocessing and n_processes > 1:
        # Use multiprocessing as in your original code
        process_func = partial(process_example, conversation=conversation)

        with Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, examples),
                total=len(examples),
                desc="Processing with Gemini"
            ))
    else:
        # Process sequentially
        results = []
        for example in tqdm(examples, desc="Processing with Gemini"):
            result = process_example(example, conversation=conversation)
            results.append(result)

    # Filter out failed results
    successful_results = [r for r in results if r is not None and r['result'] is not None]

    # Separate successful results by type
    successful_single = [r for r in successful_results if r.get('prompt_type') == 'single_stage']
    successful_evolutionary = [r for r in successful_results if r.get('prompt_type') == 'evolutionary']

    print(f"Successfully processed {len(successful_results)}/{len(examples)} examples")
    print(f"  - Single-stage success: {len(successful_single)}/{len(single_stage) if single_stage else 0}")
    print(f"  - Evolutionary success: {len(successful_evolutionary)}/{len(evolutionary) if evolutionary else 0}")

    # Save using your existing function
    save_results(successful_results, output_file)

    # Create summary with detailed statistics
    summary = {
        'metadata': {
            'total_examples': len(examples),
            'successful_results': len(successful_results),
            'single_stage_examples': len(single_stage),
            'evolutionary_examples': len(evolutionary),
            'single_stage_success': len(successful_single),
            'evolutionary_success': len(successful_evolutionary),
            'conversation_mode': conversation,
            'feature_dim': real_features.shape[1] if len(real_features.shape) > 1 else len(real_features[0]),
            'max_deviations': max_deviations,
            'min_deviations': min_deviations,
            'create_evolutionary_prompts': create_evolutionary_prompts,
            'min_stages_for_evolution': min_stages_for_evolution
        },
        'results': successful_results
    }

    # Save complete dataset
    final_output = output_file.replace('.json', '_complete.json')
    save_results(summary, final_output)

    print(f"Dataset creation complete!")
    print(f"- Results saved to: {output_file}")
    print(f"- Complete dataset saved to: {final_output}")
    print(f"- Overall success rate: {len(successful_results) / len(examples) * 100:.1f}%")
    if single_stage:
        print(f"- Single-stage success rate: {len(successful_single) / len(single_stage) * 100:.1f}%")
    else:
        print("- No single-stage examples")
    if evolutionary:
        print(f"- Evolutionary success rate: {len(successful_evolutionary) / len(evolutionary) * 100:.1f}%")
    else:
        print("- No evolutionary examples")

    return summary


# Example usage
if __name__ == "__main__":
    # Example with real data pipeline (your exact code)
    print("Creating stellar evolution dataset with evolutionary prompts...")

    # Load your real data
    latent_features = np.load('logs/2025-07-29/features.npy')
    metadata_df = pd.read_csv('logs/2025-07-29/info.csv')
    metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains('^Unnamed')]
    berger_lamost = pd.read_csv('tables/lamost_dr8_with_berger_labels.csv')
    berger_lamost = berger_lamost.loc[:, ~berger_lamost.columns.str.contains('^Unnamed')]

    real_stars_df, real_features = merge_dataframes_and_filter_array(
        berger_lamost, metadata_df, 'obsid', 'obsid', latent_features
    )
    real_stars_df.rename(columns={'Lstar': 'L'}, inplace=True)

    # Run the complete pipeline with evolutionary prompts
    result = main(
        real_stars_df=real_stars_df,
        real_features=real_features,
        n_tracks=100,
        age_points=4,
        output_file='stellar_evolution_results.json',
        conversation=False,  # Set to True for conversation mode
        use_multiprocessing=False,  # Set to True for parallel processing
        n_processes=4,
        max_deviations={
            'Teff': 100,  # ±100 K temperature difference
            'logg': 0.3,  # ±0.3 dex surface gravity difference
            'L': 1.5,  # ±1.5 solar luminosity difference
            'Mstar': 0.25  # ±0.25 solar mass difference
        },
        min_deviations={
            'Teff': 0,
            'logg': 0,
            'L': 0,
            'Mstar': 0
        },
        create_evolutionary_prompts=True,  # Enable evolutionary prompts
        min_stages_for_evolution=2  # Minimum 2 stages for evolutionary prompts
    )

    print("\nDataset composition:")
    print(f"Total examples: {result['metadata']['total_examples']}")
    print(f"Single-stage examples: {result['metadata']['single_stage_examples']}")
    print(f"Evolutionary examples: {result['metadata']['evolutionary_examples']}")

    print("\nSample results:")
    if result['results']:
        # Show single-stage example
        single_examples = [r for r in result['results'] if r.get('prompt_type') == 'single_stage']
        if single_examples:
            sample = single_examples[0]
            print(f"\nSingle-stage example (Track {sample['track_id']}):")
            print(f"Result preview: {str(sample['result'])[:150]}...")

        # Show evolutionary example
        evolutionary_examples = [r for r in result['results'] if r.get('prompt_type') == 'evolutionary']
        if evolutionary_examples:
            sample = evolutionary_examples[0]
            print(f"\nEvolutionary example (Track {sample['track_id']}):")
            print(f"Number of stages: {sample.get('n_stages', 'unknown')}")
            print(f"Result preview: {str(sample['result'])[:150]}...")
            print(f"Features shape: {np.array(sample['features']).shape}")

            # Show feature array structure for evolutionary examples
            print("\nEvolutionary example feature structure:")
            sample_features = np.array(sample['features'])
            print(f"Features array shape: {sample_features.shape}")
            print(f"Each row represents features for one evolutionary stage")

            # Show stage mapping
            if 'stage_mapping' in sample:
                print("Stage mapping:")
                for stage in sample['stage_mapping']:
                    print(f"  Stage {stage['stage_index']}: Age {stage['age']:.2f} Gyr")

    print("\nSample result:")
    if result['results']:
        sample = result['results'][0]
        print(f"Track ID: {sample['track_id']}")
        print(f"Result preview: {str(sample['result'])[:200]}...")