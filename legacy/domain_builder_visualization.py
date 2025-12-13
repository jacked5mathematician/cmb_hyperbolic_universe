import snappy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

def visualize_dirichlet_domain_with_points(manifold_name, num_points=1000):
    # Load a hyperbolic manifold
    M = snappy.Manifold(manifold_name)

    # Compute the Dirichlet domain for the manifold
    try:    
        D = M.dirichlet_domain(maximize_injectivity_radius=True)
    except RuntimeError as e:
        print(f"Failed to compute Dirichlet domain: {e}")
        return
    
    matrices = D.pairing_matrices()
    
    # Extract the vertices and faces from the Dirichlet domain
    vertex_details = D.vertex_list(details=True)
    
    # Convert vertex positions (which are tuples) to a numpy array of float type
    vertices = np.array([list(v['position']) for v in vertex_details], dtype=float)

    # Ensure the vertices are in the correct shape for plotting
    if vertices.shape[1] != 3:
        print(f"Unexpected vertex shape: {vertices.shape}")
        return

    # Extract the face data
    faces = D.face_list()

    # Use the 'hue' key to assign colors to faces that are potentially paired
    unique_hues = list(set(face['hue'] for face in faces))

    # Generate a colormap
    colors = list(mcolors.TABLEAU_COLORS.values())  # Choose from matplotlib's predefined color palette
    hue_to_color = {hue: colors[i % len(colors)] for i, hue in enumerate(unique_hues)}

    # Get the bounding box for the domain
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)

    # Generate random points within the bounding box
    points = np.random.uniform(min_corner, max_corner, (num_points, 3))

    # Filter points that are inside the domain
    inside_points = []
    for point in points:
        is_inside = True
        for face in faces:
            face_vertices = vertices[face['vertex_indices']]
            # Calculate the normal vector for the face
            if len(face_vertices) >= 3:
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                
                # Check if the point is on the correct side of the face
                if np.dot(point - face_vertices[0], normal) > 0:
                    is_inside = False
                    break
        if is_inside:
            inside_points.append(point)

    # Select a single point to plot
    if inside_points:
        point_to_plot = np.array(inside_points[0])
        print(f"Number of points found inside the domain: {len(inside_points)}")
    else:
        print("No points found inside the domain.")
        point_to_plot = None

    # Prepare the figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face of the Dirichlet domain
    for face in faces:
        # Extract the indices of vertices that make up the face
        face_vertex_indices = face['vertex_indices']
        face_vertices = vertices[face_vertex_indices]

        # Create a polygon for the face
        poly = Poly3DCollection([face_vertices], alpha=0.7)
        
        # Set face and edge color for better visualization
        poly.set_facecolor(hue_to_color[face['hue']])  # Use the color assigned to the hue
        poly.set_edgecolor('k')
        
        # Add the polygon to the plot
        ax.add_collection3d(poly)

    # Plot the selected point inside the domain
    if point_to_plot is not None:
        ax.scatter(point_to_plot[0], point_to_plot[1], point_to_plot[2], color='red', s=50, label='Test Point in Domain')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title(f"Dirichlet Domain of {manifold_name}")
    # Set the aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()

# Test the visualization with different manifolds and plot a random point
visualize_dirichlet_domain_with_points('m003(-3,1)', num_points=10000)  # Weeks manifold