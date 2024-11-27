import numpy as np
import pickle  # Import pickle to load .pkl files
from mpi4py import MPI
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    # generate_transformed_points,  # Remove this import
    convert_to_points_images,
    solve_system_via_svd_numeric,
    plot_chi_squared_spectrum,
    compute_target_M,
    filter_points_for_overconstraint,
    determine_tiling_radius,
    construct_numeric_matrix,
    poincare_to_pseudo_spherical,  # Import the transformation function
)
from tqdm import tqdm  # Import for progress bars
from joblib import Parallel, delayed
import cProfile
import pstats
import io
import time
import logging
import os
import sys
import json
from utils.sys_generation import generate_matrix_system

# Profiler function to wrap any function you want to profile
def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    result = func(*args, **kwargs)
    pr.disable()  # Stop profiling

    # Create a string buffer to hold the profile results
    s = io.StringIO()

    # Create a Stats object
    ps = pstats.Stats(pr, stream=s)

    # Sort and print by different criteria
    sort_criteria = ['cumulative', 'time', 'calls']
    for criteria in sort_criteria:
        s.write(f"\n---- Profile sorted by {criteria} ----\n")
        ps.sort_stats(criteria).print_stats(10)  # Print top 10 functions

    # Save profiling results to a file for later inspection
    with open("profiling_results.txt", "w") as f:
        f.write(s.getvalue())

    # Print the profile statistics to the console
    print(s.getvalue())  # Print the contents of the buffer to the console

    return result

def process_k_values_chunk(
    chunk_index, 
    k_values_chunk, 
    data_with_distances,  # Adjusted to receive data_with_distances
    min_images, 
    tolerance, 
    manifold_name,   
    num_best_to_compute=5
):
    # Set up logging
    output_dir = 'output_values'
    os.makedirs(output_dir, exist_ok=True)
    
    # Use chunk_index for unique log files
    logger = logging.getLogger(f"Chunk_{chunk_index}")
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(output_dir, f"chunk_{chunk_index}.log")
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set up directory for matrices
    matrices_dir = 'output_matrices'
    os.makedirs(matrices_dir, exist_ok=True)
    
    chi_squared_values_chunk = [[] for _ in range(num_best_to_compute)]
    k_values_processed = []

    # Initialize variables per chunk
    previous_c_value = None
    classified_transformed_points = None
    selected_points = None
    selected_transformed_points = None
    points_images = None

    # Process each k_value sequentially
    for k_value in k_values_chunk:
        timings = {}
        k_start_time = time.time()
        
        # Compute new c and L based on the current k_value
        c_start_time = time.time()
        new_c_value = 10 + round(100 / k_value)
        new_L_value = 10 + round(k_value)
        timings['compute_c_L'] = time.time() - c_start_time

        # Only recompute if c_value has changed
        if new_c_value != previous_c_value:
            # Compute tiling radius
            tiling_start_time = time.time()
            result = determine_tiling_radius(
                data_with_distances, new_L_value, new_c_value, min_images, tolerance
            )
            timings['determine_tiling_radius'] = time.time() - tiling_start_time

            if result is None:
                logger.error(f"Error: Could not determine tiling radius for k = {k_value}.")
                continue

            classified_transformed_points, rho_min, rho_max, valid_points = result

            # Compute M and N
            compute_M_start_time = time.time()
            M_desired, N = compute_target_M(new_L_value, new_c_value)
            timings['compute_target_M'] = time.time() - compute_M_start_time

            # Filter points
            filter_points_start_time = time.time()
            selected_points, selected_transformed_points, points_images = filter_points_for_overconstraint(
                classified_transformed_points, data_with_distances, M_desired
            )
            timings['filter_points_for_overconstraint'] = time.time() - filter_points_start_time

            previous_c_value = new_c_value

        valid_points = len(selected_points)

        # Generate matrix system
        matrix_system_start_time = time.time()
        _, _, A = generate_matrix_system(points_images, new_L_value, k_value)
        timings['generate_matrix_system'] = time.time() - matrix_system_start_time

        if A.size == 0:
            logger.error(f"Error: matrix A is empty for k = {k_value}")
            continue

        # Print matrix shape and save
        logger.info(f"Generated matrix A with shape: {A.shape}")
        matrix_filename = os.path.join(matrices_dir, f'matrix_chunk_{chunk_index}_k_{k_value:.3f}.npy')
        np.save(matrix_filename, A)
        logger.info(f"Matrix A saved to {matrix_filename}")

        # Compute SVD and chi-squared values
        chi_squared_values, singular_vectors = solve_system_via_svd_numeric(A)

        # Log the chi-squared values
        logger.info("Computed chi-squared values:")
        for idx, chi_squared in enumerate(chi_squared_values):
            logger.info(f"Singular Vector {idx + 1}: chi² = {chi_squared}")

        # Store the top chi-squared values up to `num_best_to_compute`
        for i in range(min(len(chi_squared_values), num_best_to_compute)):
            chi_squared_values_chunk[i].append(chi_squared_values[i])

        timings['total_time'] = time.time() - k_start_time

        # Log timings and chi-squared values
        logger.info(f"k = {k_value}: chi_squared_values = {chi_squared_values[:num_best_to_compute]}, timings = {timings}")

        k_values_processed.append(k_value)
    
    # Remove handler after processing
    logger.removeHandler(handler)
    handler.close()

    # Save results to a file
    output_file = os.path.join(output_dir, f"results_chunk_{chunk_index}.npz")

    # Prepare chi-squared data for saving
    chi_squared_data = {f'chi_squared_rank_{i+1}': np.array(chi_squared_values_chunk[i]) for i in range(num_best_to_compute)}

    # Save k_values and chi-squared data
    np.savez(
        output_file,
        k_values=k_values_processed,
        **chi_squared_data
    )

    logger.info(f"Results saved to {output_file}")

def main():
    # Get environment variables
    node_index = int(os.environ.get('SLURM_PROCID', '0'))
    num_nodes = int(os.environ.get('NUM_NODES', '1'))
    chunks_per_node = int(os.environ.get('CHUNKS_PER_NODE', '1'))
    num_jobs = int(os.environ.get('NUM_JOBS', '1'))  # Number of parallel jobs within each node

    # Set threading environment variables if necessary
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")

    manifold_name = 'm188(-1,1)'  # Example manifold name
    min_images = 20               # Minimum number of images required per point
    tolerance = 0.1               # Allow small deviations in rho
    resolution = 400              # Resolution for the k values
    k_values = np.linspace(1.0, 10.0, resolution)  # Range of k values

    # Total number of chunks
    total_chunks = num_nodes * chunks_per_node

    # Split k_values into chunks using round-robin
    k_values_chunks = [[] for _ in range(total_chunks)]
    for idx, k_value in enumerate(k_values):
        chunk_idx = idx % total_chunks  # Round-robin assignment
        k_values_chunks[chunk_idx].append(k_value)

    # Assign chunks to this node
    chunks_assigned = []
    chunk_indices = []
    for i in range(total_chunks):
        if i % num_nodes == node_index:
            chunks_assigned.append(k_values_chunks[i])
            chunk_indices.append(i)

    # Load transformed points data from the .pkl file
    transformed_data_file = f'{manifold_name}_points_data.pkl'  # Update file name

    if os.path.exists(transformed_data_file):
        print(f"Loading transformed points data from {transformed_data_file}...")
        with open(transformed_data_file, 'rb') as f:
            data_with_distances = pickle.load(f)
        print(f"Loaded transformed points data.")
    else:
        print(f"Error: {transformed_data_file} not found.")
        return

    # Process the assigned chunks in parallel
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_k_values_chunk)(
            chunk_index, k_values_chunk, data_with_distances,
            min_images, tolerance, manifold_name,
        ) for chunk_index, k_values_chunk in zip(chunk_indices, chunks_assigned)
    )

    config = {
    "total_chunks": total_chunks, 
    "resolution": resolution,
    "manifold_name": manifold_name
    }   

    with open('output_values/config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    # Initialize logging at the beginning
    logging.basicConfig(level=logging.INFO)
    profile_function(main)