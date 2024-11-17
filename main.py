import numpy as np
from mpi4py import MPI
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    generate_transformed_points,
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
from joblib import Parallel, delayed
import cProfile
import pstats
import io
import time
import logging
import os
import sys
import json

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
    inside_points, 
    pairing_matrices, 
    min_images, 
    tolerance, 
    manifold_name, 
    num_best_to_compute=5  # Number of best chi-squared values to compute
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
                inside_points, pairing_matrices, new_L_value, new_c_value, min_images, tolerance
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
            selected_points, selected_transformed_points = filter_points_for_overconstraint(
                classified_transformed_points, inside_points, M_desired
            )
            timings['filter_points_for_overconstraint'] = time.time() - filter_points_start_time

            # Convert points
            convert_points_start_time = time.time()
            points_images = convert_to_points_images(selected_transformed_points)
            timings['convert_to_points_images'] = time.time() - convert_points_start_time

            previous_c_value = new_c_value

        valid_points = len(selected_points)

        # Generate matrix system
        matrix_system_start_time = time.time()
        _, _, matrix_system = generate_matrix_system(points_images, new_L_value, k_value, valid_points)
        timings['generate_matrix_system'] = time.time() - matrix_system_start_time

        if len(matrix_system) == 0 or len(matrix_system[0]) == 0:
            logger.error(f"Error: matrix_system is empty for k = {k_value}")
            continue

        # Construct numeric matrix
        construct_matrix_start_time = time.time()
        A = construct_numeric_matrix(matrix_system, k_value)
        timings['construct_numeric_matrix'] = time.time() - construct_matrix_start_time

        # Save the matrix A
        matrix_filename = os.path.join(matrices_dir, f'matrix_chunk_{chunk_index}_k_{k_value:.3f}.npz')
        np.savez_compressed(matrix_filename, A=A)
        logger.info(f"Matrix A saved to {matrix_filename}")

        # Solve system via SVD
        solve_system_start_time = time.time()
        chi_squared_values, _ = solve_system_via_svd_numeric(A)
        timings['solve_system_via_svd_numeric'] = time.time() - solve_system_start_time

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
    num_points = 10000            # Number of random points to generate
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

    # Build Dirichlet domain
    domain_data = build_dirichlet_domain(manifold_name)
    #if domain_data is None:
        #print("Failed to build Dirichlet domain.")
        #return
    vertices, faces, pairing_matrices = domain_data

    # Generate random points
    #points = generate_random_points_in_domain(vertices, num_points)
    output_dir = 'output_values'
    inside_points_file = os.path.join(output_dir, 'inside_points.npy')

    # Filter points inside the domain
    # Check if inside_points file exists
    if os.path.exists(inside_points_file):
        print("Loading inside_points from file...")
        inside_points = np.load(inside_points_file)
        print(f"Loaded {len(inside_points)} points from inside_points file.")
    else:
        print("Error: inside_points file not found. Please run the debug file to generate inside_points.npy.")
        return
    print(f"Number of points found inside the domain: {len(inside_points)}")

    # Process the assigned chunks in parallel
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_k_values_chunk)(
            chunk_index, k_values_chunk, inside_points, pairing_matrices,
            min_images, tolerance, manifold_name
        ) for chunk_index, k_values_chunk in zip(chunk_indices, chunks_assigned)
    )
    config = {
        "total_chunks": total_chunks, 
        "resolution": resolution       
    }

    with open('output_values/config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    # Initialize logging at the beginning
    logging.basicConfig(level=logging.INFO)
    profile_function(main)