import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import generate_transformed_points, build_dirichlet_domain, generate_random_points_in_domain, filter_points_in_domain, klein_to_pseudo_spherical# Assuming this is where the function is


def plot_image_distribution_vs_rho_max(inside_points, pairing_matrices, rho_max_values):
    """
    Plot the distribution of the number of images for each point as rho_max varies.
    Also, plot the 2D projection of the point images in polar coordinates (rho, theta).
    Finally, plot the original 3 points and their images in the covering space.
    
    Parameters:
    - inside_points: List of points inside the domain.
    - pairing_matrices: Pairing matrices to apply transformations.
    - rho_max_values: List or array of rho_max values to iterate through.
    """

    # To hold the results for each rho_max
    image_counts_per_rho_max = []
    all_images_rho_theta = []

    # Iterate through the range of rho_max values
    for rho_max in tqdm(rho_max_values, desc="Processing rho_max values"):
        # Generate the transformed points for the current rho_max
        classified_transformed_points = generate_transformed_points(inside_points, pairing_matrices, rho_max)
        
        # Get the number of images for each point
        image_counts = [len(images) for images in classified_transformed_points.values()]
        
        # Store the result for this rho_max
        image_counts_per_rho_max.append(image_counts)
        
        # Store the rho and theta values for plotting
        for images in classified_transformed_points.values():
            for image in images:
                rho, theta = image[:2]  # We are assuming pseudo-spherical coordinates (rho, theta)
                all_images_rho_theta.append((rho, theta))

    # Plot histograms for the distributions
    plt.figure(figsize=(10, 6))
    
    for i, rho_max in enumerate(rho_max_values):
        plt.hist(image_counts_per_rho_max[i], bins=20, alpha=0.5, label=f'rho_max = {rho_max}')
    
    plt.title('Distribution of Number of Images per Point for Different rho_max Values')
    plt.xlabel('Number of Images')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the images in the covering space (polar coordinates: rho vs theta)
    plt.figure(figsize=(8, 8))
    
    # Scatter plot of the points' images in the (rho, theta) plane
    all_images_rho_theta = np.array(all_images_rho_theta)
    plt.polar(all_images_rho_theta[:, 1], all_images_rho_theta[:, 0], 'bo', markersize=3, label='Images')  # theta, rho

    # Add circular boundaries for different rho_max values
    for rho_max in rho_max_values:
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(theta, np.full_like(theta, rho_max), linestyle='--', alpha=0.7, label=f'rho_max = {rho_max}')

    plt.title('Images of Points in Polar Coordinates (rho vs theta)')
    plt.legend()
    plt.show()

    # THIRD PLOT: Original points and their images in different colors
    # Select 3 random points from the inside_points (fundamental domain)
    selected_indices = np.random.choice(len(inside_points), 3, replace=False)
    selected_points = [inside_points[i] for i in selected_indices]

    # Prepare the color map for original points and their images
    colors = ['r', 'g', 'b']  # One color per point

    plt.figure(figsize=(8, 8))

    # Iterate through each selected point
    for i, point in enumerate(selected_points):
        # Convert the original point from Klein to pseudo-spherical coordinates
        original_rho_theta = klein_to_pseudo_spherical([point])[0]
        
        # Plot the original point in polar coordinates
        plt.polar(original_rho_theta[1], original_rho_theta[0], color=colors[i], marker='o', markersize=10, label=f'Original Point {i+1}')
        
        # Generate the transformed images for this point
        classified_transformed_points = generate_transformed_points([point], pairing_matrices, rho_max_values[-1])  # Use the largest rho_max
        
        # Plot the transformed images
        for images in classified_transformed_points.values():
            for image in images:
                rho, theta = image[:2]  # Extract rho and theta
                plt.polar(theta, rho, color=colors[i], marker='x', markersize=5, alpha=0.7)

    plt.title('Original Points and Their Images in Polar Coordinates')
    plt.legend()
    plt.show()

# Example usage:
if __name__ == "__main__":
    manifold_name = 'm188(-1,1)'  # Example manifold name
    num_points = 1000  # Number of random points to generate
    rho_max_values = np.linspace(0.5, 5.0, 10)  # Vary rho_max from 0.5 to 5.0 in 10 steps

    # Step 1: Build Dirichlet domain
    domain_data = build_dirichlet_domain(manifold_name)
    if domain_data is None:
        print("Failed to build Dirichlet domain.")
    vertices, faces, pairing_matrices = domain_data

    # Step 2: Generate random points
    points = generate_random_points_in_domain(vertices, num_points)

    # Step 3: Filter points inside the domain
    inside_points = filter_points_in_domain(points, faces, vertices)

    # Plot the distribution and covering space projection
    plot_image_distribution_vs_rho_max(inside_points, pairing_matrices, rho_max_values)