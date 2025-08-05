import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator
import pandas as pd
import kiauhoku as kh
from tqdm import tqdm
import time
import scipy.stats as stats

def giant_cond(x):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :param: x row in dataframe with columns - Teff, logg
    :return: boolean
    """
    logg, teff = x['logg'], x['Teff']
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh



def get_yrec_params(grid_name, mass, feh, alpha, age):
    df = kh.load_full_grid(grid_name)
    # Parameters to interpolate
    params_to_interpolate = ['Log Teff(K)', 'logg', 'L/Lsun', 'R/Rsun', 'Prot(days)']

    # Convert inputs to arrays if they're not already
    mass_arr = np.atleast_1d(mass)
    feh_arr = np.atleast_1d(feh)
    alpha_arr = np.atleast_1d(alpha)
    age_arr = np.atleast_1d(age)

    results = {param: np.zeros_like(mass_arr, dtype=float) for param in params_to_interpolate}

    # Create the points array directly from the DataFrame index and Age column
    print("Preparing data points...")
    start_time = time.time()

    # Directly extract m, f, a values from MultiIndex
    m_values = df.index.get_level_values(0).values
    f_values = df.index.get_level_values(1).values
    a_values = df.index.get_level_values(2).values
    age_values = df['Age(Gyr)'].values

    # Combine coordinates into points array
    points = np.column_stack([m_values, f_values, a_values, age_values])
    query_points = np.column_stack([mass_arr, feh_arr, alpha_arr, age_arr])
    return points, query_points, params_to_interpolate, df


def get_mist_params(grid_name, mass, feh, age):
    df = kh.load_grid(grid_name)
    # Parameters to interpolate
    params_to_interpolate = ['log_Teff', 'log_g', 'log_L', 'log_R']

    # Convert inputs to arrays if they're not already
    mass_arr = np.atleast_1d(mass)
    feh_arr = np.atleast_1d(feh)
    age_arr = np.atleast_1d(age)

    # Create the points array directly from the DataFrame index and Age column
    print("Preparing data points...")
    start_time = time.time()

    # Directly extract m, f, a values from MultiIndex
    m_values = df.index.get_level_values(0).values
    f_values = df.index.get_level_values(1).values
    age_values = df['star_age'].values / 1e9

    # Combine coordinates into points array
    points = np.column_stack([m_values, f_values, age_values])
    query_points = np.column_stack([mass_arr, feh_arr, age_arr])
    return points, query_points, params_to_interpolate, df


def interpolate_stellar_parameters(mass, feh, alpha, age, grid_name='fastlaunch', method='nearest'):
    """
    Interpolate stellar parameters using values from the full grid.

    Parameters:
    -----------
    mass : float or array-like
        Mass value(s) to interpolate for
    feh : float or array-like
        [Fe/H] value(s) to interpolate for
    alpha : float or array-like
        [alpha/Fe] value(s) to interpolate for
    age : float or array-like
        Age value(s) to interpolate for
    method : str, optional
        Interpolation method: 'nearest', 'linear', 'rbf', or 'kd_tree'

    Returns:
    --------
    dict of interpolated values for Teff, logg, L/Lsun, and R/Rsun
    """
    start_time = time.time()
    # Load the grid directly
    print("Loading grid...")
    if grid_name != 'mist':
        points, query_points, params_to_interpolate, df = get_yrec_params(grid_name, mass, feh, alpha, age)
    else:
        points, query_points, params_to_interpolate, df = get_mist_params(grid_name, mass, feh, age)

    results = {param: np.zeros_like(mass, dtype=float) for param in params_to_interpolate}

    print(f"Points preparation took {time.time() - start_time:.2f} seconds")

    # Prepare query points

    # Select interpolation method
    interpolator_class = get_interpolator(method)

    # Use the selected interpolation method
    for param in params_to_interpolate:
        print(f"Interpolating {param}...")
        start_time = time.time()

        # Extract values directly
        values = df[param].values

        # Create interpolator
        if method == 'kd_tree':
            interp = interpolator_class(points, values[:, np.newaxis])
            results[param] = interp(query_points)[:, 0]
        else:
            interp = interpolator_class(points, values)
            results[param] = interp(query_points)

        print(f"Interpolation of {param} took {time.time() - start_time:.2f} seconds")

    if grid_name != 'mist':
        teff = 10 ** results['Log Teff(K)']
        logg = results['logg']
        L = results['L/Lsun']
        R = results['R/Rsun']
        Prot = results['Prot(days)']
    else:
        teff = 10 ** results['log_Teff']
        logg = results['log_g']
        L = 10 ** results['log_L']
        R = 10 ** results['log_R']
        Prot = None
    return {'Teff': teff, 'logg': logg, 'L': L, 'R': R, 'Prot': Prot}


def get_interpolator(method):
    if method == 'nearest':
        # Nearest neighbor interpolation (fastest)
        from scipy.interpolate import NearestNDInterpolator
        interpolator_class = NearestNDInterpolator
    elif method == 'linear':
        # Linear interpolation (slower but more accurate)
        from scipy.interpolate import LinearNDInterpolator
        interpolator_class = LinearNDInterpolator
    elif method == 'rbf':
        # Radial basis function interpolation (may be faster for large datasets)
        from scipy.interpolate import RBFInterpolator
        def create_interp(pts, vals):
            return RBFInterpolator(pts, vals, neighbors=10, kernel='thin_plate_spline')

        interpolator_class = create_interp
    elif method == 'kd_tree':
        # Custom KD-tree based interpolation
        from scipy.spatial import cKDTree

        def create_kdtree_interp(pts, vals):
            tree = cKDTree(pts)

            def interpolate_func(query_pts):
                # Find k nearest neighbors
                distances, indices = tree.query(query_pts, k=8)

                # Compute weights (inverse distance weighting)
                weights = 1.0 / (distances + 1e-10)  # Add small constant to avoid division by zero
                weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

                # Compute weighted average
                result = np.sum(vals[indices] * weights[:, :, np.newaxis], axis=1)
                return result

            return interpolate_func

        interpolator_class = create_kdtree_interp
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    return interpolator_class


def truncated_normal_dist(mean, std, lower_bound, upper_bound, size):
    a = (lower_bound - mean) / std
    b = (upper_bound - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def kroupa_imf(m):
    """
    Kroupa Initial Mass Function

    Parameters:
    m : array-like
        Stellar masses in solar mass units

    Returns:
    xi : array-like
        Number of stars per mass interval
    """
    # Define the exponents for different mass ranges
    alpha_1 = 1.3  # For m in [0.08, 0.5) solar masses
    alpha_2 = 2.3  # For m in [0.5, infinity) solar masses

    # Initialize the output array
    xi = np.zeros_like(m, dtype=float)

    # Calculate IMF for different mass ranges
    # Note: We implement both ranges even though our focus is 0.3-2 solar masses
    mask_low = (m >= 0.08) & (m < 0.5)
    mask_high = m >= 0.5

    # Set the normalization at 0.5 solar masses
    m_0 = 0.5

    # Calculate IMF values
    xi[mask_low] = m[mask_low] ** (-alpha_1)
    xi[mask_high] = m_0 ** (alpha_2 - alpha_1) * m[mask_high] ** (-alpha_2)

    return xi


# Create a function to sample from this distribution
def sample_kroupa_imf(n_samples, m_min=0.3, m_max=2.0):
    """
    Generate random samples from the Kroupa IMF

    Parameters:
    n_samples : int
        Number of samples to generate
    m_min : float
        Minimum mass in solar masses
    m_max : float
        Maximum mass in solar masses

    Returns:
    masses : ndarray
        Array of sampled masses
    """
    # Create a fine grid for numerical integration and interpolation
    m_grid = np.logspace(np.log10(m_min), np.log10(m_max), 1000)
    imf_values = kroupa_imf(m_grid)

    # Create cumulative distribution function
    cdf = np.cumsum(imf_values)
    cdf = cdf / cdf[-1]  # Normalize

    # Draw random samples from uniform distribution
    u = np.random.random(n_samples)

    # Interpolate to find the corresponding masses
    masses = np.interp(u, cdf, m_grid)

    return masses


def generate_evolutionary_tracks(n_stars, age_points, m_min=0.5, m_max=2,
                                 min_age=0.1, max_age=10, grid_name='fastlaunch',
                                 method='nearest', feh_range=None):
    """
    Generate evolutionary tracks for multiple stars with solar metallicity.

    Parameters:
    -----------
    n_stars : int
        Number of stars to generate tracks for
    age_points : int or array-like
        If int: number of age points to sample logarithmically between min_age and max_age
        If array: specific ages (in Gyr) to compute for each star
    m_min, m_max : float
        Mass range for sampling from Kroupa IMF
    min_age, max_age : float
        Age range in Gyr (only used if age_points is int)
    grid_name : str
        Stellar evolution grid to use
    method : str
        Interpolation method

    Returns:
    --------
    dict containing:
        - 'star_id': array of star IDs for each data point
        - 'mass': array of masses for each star (constant along track)
        - 'age': array of ages
        - 'Teff': array of effective temperatures
        - 'logg': array of surface gravities
        - 'L': array of luminosities
        - 'R': array of radii
        - 'Prot': array of rotation periods (if available)
    """

    # Sample stellar masses from Kroupa IMF
    stellar_masses = sample_kroupa_imf(n_stars, m_min=m_min, m_max=m_max)

    # All stars have solar metallicity and alpha abundance
    if feh_range is None:
        stellar_feh = np.zeros(n_stars)  # [Fe/H] = 0 (solar)
    else:
        stellar_feh = np.linspace(feh_range[0], feh_range[1], n_stars)
    stellar_alpha = np.zeros(n_stars)  # [alpha/Fe] = 0 (solar)

    # Generate age array
    if isinstance(age_points, int):
        ages = np.logspace(np.log10(min_age), np.log10(max_age), age_points)
    else:
        ages = np.array(age_points)

    n_ages = len(ages)
    total_points = n_stars * n_ages

    # Create arrays for all combinations of stars and ages
    star_ids = np.repeat(np.arange(n_stars), n_ages)
    masses = np.repeat(stellar_masses, n_ages)
    feh_values = np.repeat(stellar_feh, n_ages)
    alpha_values = np.repeat(stellar_alpha, n_ages)
    age_values = np.tile(ages, n_stars)

    print(f"Computing evolutionary tracks for {n_stars} solar metallicity stars at {n_ages} age points...")
    print(f"Mass range: {stellar_masses.min():.3f} - {stellar_masses.max():.3f} M☉")
    print(f"Total interpolation points: {total_points}")

    # Interpolate all parameters at once
    results = interpolate_stellar_parameters(
        masses, feh_values, alpha_values, age_values,
        grid_name=grid_name, method=method
    )

    # Compile final results (simplified - no need to store feh/alpha since they're all zero)
    tracks = {
        'star_id': star_ids,
        'mass': masses,
        'age': age_values,
        'feh': feh_values,
        'Teff': results['Teff'],
        'logg': results['logg'],
        'L': results['L'],
        'R': results['R'],
        'Prot': results['Prot']
    }

    return tracks


def tracks_to_dataframe(tracks):
    """
    Convert evolutionary tracks dictionary to a pandas DataFrame.
    """
    return pd.DataFrame(tracks)


def get_single_star_track(tracks, star_id):
    """
    Extract the evolutionary track for a single star.

    Parameters:
    -----------
    tracks : dict
        Output from generate_evolutionary_tracks
    star_id : int
        ID of the star to extract

    Returns:
    --------
    dict with the evolutionary track for the specified star
    """
    mask = tracks['star_id'] == star_id
    return {key: val[mask] for key, val in tracks.items()}


# Example usage and plotting function
def plot_evolutionary_tracks(tracks, n_plot=5, parameters=['Teff', 'L']):
    """
    Plot evolutionary tracks for a subset of stars.

    Parameters:
    -----------
    tracks : dict
        Output from generate_evolutionary_tracks
    n_plot : int
        Number of stars to plot
    parameters : list
        List of parameters to plot (e.g., ['Teff', 'L'] for HR diagram)
    """
    import matplotlib.pyplot as plt

    unique_stars = np.unique(tracks['star_id'])[:n_plot]

    fig, ax = plt.subplots(figsize=(10, 8))

    for star_id in unique_stars:
        star_track = get_single_star_track(tracks, star_id)
        mass = star_track['mass'][0]  # Mass is constant for each star

        if len(parameters) == 2:
            x_param, y_param = parameters
            ax.plot(star_track[x_param], star_track[y_param],
                    label=f'M={mass:.2f} M☉', alpha=0.7, marker='o', markersize=3)

    ax.set_xlabel(parameters[0])
    ax.set_ylabel(parameters[1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    if parameters == ['Teff', 'L']:
        ax.invert_xaxis()  # HR diagram convention
        ax.set_yscale('log')
        ax.set_xlabel('Effective Temperature (K)')
        ax.set_ylabel('Luminosity (L☉)')
        ax.set_title('Hertzsprung-Russell Diagram')

    plt.tight_layout()
    plt.savefig('figs/evolutionary_tracks.png')
    plt.show()


if __name__ == '__main__':
    # Example 1: Solar metallicity stars
    print("=== Example 1: Solar metallicity stars ===")
    tracks_solar = generate_evolutionary_tracks(
        n_stars=50,
        age_points=20,
        m_min=0.5,
        m_max=2.0,
        min_age=0.1,
        max_age=10.0,
        grid_name='fastlaunch',
        method='nearest'
    )

    # Example 2: Variable metallicity stars
    print("\n=== Example 2: Variable metallicity stars ===")
    tracks_variable = generate_evolutionary_tracks(
        n_stars=5000,
        age_points=4,
        m_min=0.5,
        m_max=4,
        min_age=1,
        max_age=10.0,
        feh_range=(-0.5, 0.3),  # From metal-poor to metal-rich
        grid_name='fastlaunch',
        method='nearest'
    )

    # Convert to DataFrames for analysis
    df = tracks_to_dataframe(tracks_variable)

    df['main_seq'] = df.apply(giant_cond, axis=1)

    print(f"Variable metallicity tracks shape: {df.shape}")

