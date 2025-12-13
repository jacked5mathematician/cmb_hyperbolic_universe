import numpy as np
import pickle  # Import pickle to load .pkl files
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    convert_to_points_images,
    solve_system_via_svd_numeric,
    plot_chi_squared_spectrum,
    compute_target_M,
    filter_points_for_overconstraint,
    determine_tiling_radius,
    generate_matrix_system,
    construct_numeric_matrix,
)
from tqdm import tqdm  # Import for progress bars
from tqdm_joblib import tqdm_joblib  # For progress bars with joblib
from joblib import Parallel, delayed
import cProfile
import pstats
import io
import time
import logging
import os
import json


def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()  
    result = func(*args, **kwargs)
    pr.disable()  
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)

    # Sort and print by different criteria
    sort_criteria = ['cumulative', 'time', 'calls']
    for criteria in sort_criteria:
        s.write(f"\n---- Profile sorted by {criteria} ----\n")
        ps.sort_stats(criteria).print_stats(10) 

    with open("profiling_results.txt", "w") as f:
        f.write(s.getvalue())

    print(s.getvalue())  

    return result

def process_k_values_chunk(process_index, k_values_chunk, data_with_distances, min_images, tolerance, manifold_name, num_chunks):

    output_dir = 'output_values_2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = logging.getLogger(f"Process_{process_index}")
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(output_dir, f"process_{process_index}.log")
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    chi_squared_values_chunk = []
    k_values_processed = []

    # Initialize variables per process
    previous_c_value = None
    classified_transformed_points = None
    rho_min, rho_max = None, None
    selected_points = None
    selected_transformed_points = None
    points_images = None

    # Unique description for each process
    process_desc = f"Chunk {process_index+1}/{num_chunks}"

    # Initialize tqdm with position to prevent overlapping
    with tqdm(k_values_chunk, desc=process_desc, position=process_index, leave=False) as pbar:
        for k_value in pbar:
            timings = {}  # Dictionary to store timing information for this k_value
            k_start_time = time.time() 

            # Compute new c and L based on the current k_value
            c_start_time = time.time()
            new_c_value = 10 + round(100 / k_value)
            new_L_value = 10 + round(k_value)
            timings['compute_c_L'] = time.time() - c_start_time

            # Only recompute the tiling radius if the value of c has changed
            if new_c_value != previous_c_value:
                # Compute the tiling radius for the new value of c
                tiling_start_time = time.time()
                result = determine_tiling_radius(
                    data_with_distances, new_L_value, new_c_value, min_images, tolerance
                )
                timings['determine_tiling_radius'] = time.time() - tiling_start_time

                if result is None:
                    logger.error(f"Error: Could not determine tiling radius for k = {k_value}.")
                    continue

                classified_transformed_points, rho_min, rho_max, valid_points = result

                # Calculate the target number of rows (M) based on the degree of over-constraint (c) and L
                compute_M_start_time = time.time()
                M_desired, N = compute_target_M(new_L_value, new_c_value)
                timings['compute_target_M'] = time.time() - compute_M_start_time

                # Filter and select points for the desired over-constraint
                filter_points_start_time = time.time()
                selected_points, selected_transformed_points, points_images = filter_points_for_overconstraint(
                    classified_transformed_points, data_with_distances, M_desired
                )
                timings['filter_points_for_overconstraint'] = time.time() - filter_points_start_time

                # Update the previous c value to the new one
                previous_c_value = new_c_value

            valid_points = len(selected_points)

            # Compute matrix system and chi-squared values using filtered matrix system
            matrix_system_start_time = time.time()
            _, _, A = generate_matrix_system(points_images, new_L_value, k_value)
            timings['generate_matrix_system'] = time.time() - matrix_system_start_time

            if A.size == 0:
                logger.error(f"Error: matrix A is empty for k = {k_value}")
                continue
            
            matrix_filename = os.path.join('output_matrices_2', f'matrix_chunk_{process_index}_k_{k_value:.3f}.npy')
            np.save(matrix_filename, A)
        
            # Solve the system via SVD
            solve_system_start_time = time.time()
            chi_squared_values, _ = solve_system_via_svd_numeric(A)
            timings['solve_system_via_svd_numeric'] = time.time() - solve_system_start_time

            # Total time for this k_value
            timings['total_time'] = time.time() - k_start_time

            logger.info(f"k = {k_value}: chi_squared = {chi_squared_values}, timings = {timings}")

            chi_squared_values_chunk.append(chi_squared_values)
            k_values_processed.append(k_value)

    logger.removeHandler(handler)
    handler.close()

    return chi_squared_values_chunk, k_values_processed

def main():
    manifold_name = 'm188(-1,1)' 
    min_images = 14  # Minimum number of images required per point
    tolerance = 0  # Allow small deviations
    resolution = 400  # Resolution for the k values
    k_values = np.linspace(1.0, 10.0, resolution)  
    
    num_chunks = 100  # Number of chunks to process in parallel

    # Distribute k_values into chunks in a round-robin fashion to balance computational load
    k_values_chunks = [[] for _ in range(num_chunks)]
    for index, k_value in enumerate(k_values):
        chunk_index = index % num_chunks
        k_values_chunks[chunk_index].append(k_value)

    # Load transformed points data from the .pkl file
    transformed_data_file = os.path.join('point_data', f'{manifold_name}_points_data.pkl') 

    if os.path.exists(transformed_data_file):
        print(f"Loading transformed points data from {transformed_data_file}...")
        with open(transformed_data_file, 'rb') as f:
            data_with_distances = pickle.load(f)
        print(f"Loaded transformed points data.")
    else:
        print(f"Error: {transformed_data_file} not found.")
        return

    # Initialize chi_squared_values as a list of three empty lists
    chi_squared_values = [[], [], []]
    k_values_collected = []

    # Initialize the progress bar
    with tqdm_joblib(tqdm(desc="Processing Chunks", total=num_chunks)) as progress_bar:
        # Parallel processing of chunks
        results = Parallel(n_jobs=-1)(
            delayed(process_k_values_chunk)(
                process_index, k_values_chunk, data_with_distances,
                min_images, tolerance, manifold_name, num_chunks
            )
            for process_index, k_values_chunk in enumerate(k_values_chunks)
        )

    for chi_squared_chunk, k_values_chunk_processed in results:
        k_values_collected.extend(k_values_chunk_processed)
        for i in range(3):
            chi_squared_values[i].extend([chi_vals[i] for chi_vals in chi_squared_chunk])

    # Sort the results by k_values
    sorted_indices = np.argsort(k_values_collected)
    k_values_sorted = np.array(k_values_collected)[sorted_indices]
    chi_squared_values_sorted = [np.array(chi_list)[sorted_indices] for chi_list in chi_squared_values]

    # Save combined results
    output_dir = 'output_values_2'
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "total_chunks": num_chunks,
        "resolution": resolution,
        "manifold_name": manifold_name
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Plot the chi-squared spectrum
    plot_chi_squared_spectrum(k_values_sorted, chi_squared_values_sorted, manifold_name, resolution)

if __name__ == "__main__":
    # Initialize logging at the beginning
    logging.basicConfig(level=logging.INFO)
    profile_function(main)
