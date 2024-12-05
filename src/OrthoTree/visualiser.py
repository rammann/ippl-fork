import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data_folder = "output"

def read_data_for_all_processors(data_folder):
    processor_data = []
    num_processors = len([f for f in os.listdir(data_folder) if f.startswith("octants") and f.endswith(".txt")])

    for i in range(num_processors):
        octants_file = os.path.join(data_folder, f"octants{i}.txt")
        particles_file = os.path.join(data_folder, f"particles{i}.txt")
        
        if not os.path.isfile(octants_file) or not os.path.isfile(particles_file):
            print(f"Warning: Missing file(s) for processor {i}")
            continue

        octants = read_octants_file(octants_file)
        particles = read_particles_file(particles_file)
        dim = len(particles[0]["coord"]) if particles else 2  # Default to 2D if no particles
        # print("OCTANTS:", octants)
        # print("PARTICLES:", particles)
        # print("DIM:", dim)
        processor_data.append((octants, particles, dim))

    return processor_data

# Reads a coordinate in the form (x, y, z) or (x, y)
def read_coordinate(coord_str):
    coords = coord_str.strip("()").split(", ")
    if len(coords) == 2:
        return float(coords[0]), float(coords[1])  # Return 2D coordinate
    elif len(coords) == 3:
        return float(coords[0]), float(coords[1]), float(coords[2])  # Return 3D coordinate
    else:
        raise ValueError("Unexpected coordinate format")

def read_octant_line(line):
    parts = line.split(' ')
    morton_code = int(parts[0])
    if len(parts) >= 14:  # 3D case with (x, y, z) coordinates
        coord_min = read_coordinate(f"({parts[2]}, {parts[4]}, {parts[6]})")
        coord_max = read_coordinate(f"({parts[9]}, {parts[11]}, {parts[13]})")
    elif len(parts) >= 10:  # 2D case with (x, y) coordinates
        coord_min = read_coordinate(f"({parts[2]}, {parts[4]})")
        coord_max = read_coordinate(f"({parts[7]}, {parts[9]})")
    else:   
        raise ValueError("Unexpected format in octant line: insufficient data")

    return {
        "morton_code": morton_code,
        "coord_min": coord_min,
        "coord_max": coord_max
    }

def read_octants_file(file_path):
    octants = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            octant = read_octant_line(line)
            octants.append(octant)
    return octants

def read_particle_line(line):
    parts = line.split(' ')
    morton_code = int(parts[0])
    if len(parts) >= 7:
        coord = read_coordinate(f"({parts[2]}, {parts[4]}, {parts[6]})")
    elif len(parts) >= 4:
        coord = read_coordinate(f"({parts[2]}, {parts[4]})")
    else:
        raise ValueError("Unexpected format in particle line: insufficient data")
    
    return {
        "morton_code": morton_code,
        "coord": coord
    }

def read_particles_file(file_path):
    particles = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            particle = read_particle_line(line)
            particles.append(particle)
    return particles

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_bounds(ax, coord_min, coord_max, morton_code=None, color='blue', alpha=0.3, fill=True):
    line_width = 1
    if len(coord_min) == 2:  # 2D case
        x_min, y_min = coord_min
        x_max, y_max = coord_max
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color=color, alpha=alpha, 
                             fill=fill, linewidth=line_width)
        ax.add_patch(rect)
        return rect, f"Octant ID: {morton_code}\nMin: {coord_min}\nMax: {coord_max}"
    
    elif len(coord_min) == 3:  # 3D case
        x_min, y_min, z_min = coord_min
        x_max, y_max, z_max = coord_max
        # Define vertices of the bounding box
        vertices = [
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min)],  # Bottom face
            [(x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max)],  # Top face
            [(x_min, y_min, z_min), (x_min, y_max, z_min), (x_min, y_max, z_max), (x_min, y_min, z_max)],  # Left face
            [(x_max, y_min, z_min), (x_max, y_max, z_min), (x_max, y_max, z_max), (x_max, y_min, z_max)],  # Right face
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_min, z_max), (x_min, y_min, z_max)],  # Front face
            [(x_min, y_max, z_min), (x_max, y_max, z_min), (x_max, y_max, z_max), (x_min, y_max, z_max)],  # Back face
        ]
        if fill:
            # Use Poly3DCollection to fill the faces of the 3D box
            poly3d_collection = Poly3DCollection(vertices, color=color, alpha=alpha, linewidths=line_width, edgecolors=color)
            ax.add_collection3d(poly3d_collection)
            return poly3d_collection, f"Octant ID: {morton_code}\nMin: {coord_min}\nMax: {coord_max}"
        else:
            # Draw the bounding lines if not filled
            lines = []
            lines += ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 
                             [z_min, z_min, z_min, z_min, z_min], color=color, alpha=alpha, linewidth=line_width)
            lines += ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 
                             [z_max, z_max, z_max, z_max, z_max], color=color, alpha=alpha, linewidth=line_width)
            for i in range(2):
                lines += ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], color=color, alpha=alpha, linewidth=line_width)
                lines += ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], color=color, alpha=alpha, linewidth=line_width)
                lines += ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], color=color, alpha=alpha, linewidth=line_width)
                lines += ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], color=color, alpha=alpha, linewidth=line_width)
            return lines[0], f"Octant ID: {morton_code}\nMin: {coord_min}\nMax: {coord_max}"

def plot_all_processors_in_subplots(data_folder):
    processor_data = read_data_for_all_processors(data_folder)
    num_processors = len(processor_data)
    cols = 2  # Define columns for the grid layout
    rows = (num_processors + 1) // cols  # Calculate required rows

    fig = plt.figure(figsize=(10, 5 * rows))

    for i, (octants, particles, dim) in enumerate(processor_data):
        print(f"Processor {i}: {len(octants)} octants, {len(particles)} particles, Dimension: {dim}D")  # Debug output

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d' if dim == 3 else None)
        ax.set_title(f"Processor {i}")
        
        # Draw octants
        for octant in octants:
            # Pass fill=True here to ensure the rectangles are filled
            draw_bounds(ax, octant["coord_min"], octant["coord_max"], morton_code=octant["morton_code"], color='blue', alpha=0.2, fill=False)

        particle_size = 0.05  # Increased size for visibility
        # Draw particles
        if dim == 2:
            x_vals = [particle["coord"][0] for particle in particles]
            y_vals = [particle["coord"][1] for particle in particles]
            ax.scatter(x_vals, y_vals, color='red', s=particle_size)
        elif dim == 3:
            x_vals = [particle["coord"][0] for particle in particles]
            y_vals = [particle["coord"][1] for particle in particles]
            z_vals = [particle["coord"][2] for particle in particles]
            ax.scatter(x_vals, y_vals, z_vals, color='red', s=particle_size)

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if dim == 3:
            ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()

import itertools

def plot_combined_processors(data_folder):
    processor_data = read_data_for_all_processors(data_folder)
    dim = 2  # Default to 2D in case no particles are found

    # Define a color cycle for different processors
    colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)
    ax.set_title("Combined Data from All Processors")
    
    for i, (octants, particles, processor_dim) in enumerate(processor_data):
        color = next(colors)  # Get the next color for this processor
        dim = max(dim, processor_dim)  # Update to 3D if any processor has 3D data

        # Draw octants with the processor-specific color
        for octant in octants:
            draw_bounds(ax, octant["coord_min"], octant["coord_max"], morton_code=octant["morton_code"], fill=False)

        # Draw particles with the processor-specific color
        particle_size = 0.05  # Adjust size if needed
        if dim == 2:
            x_vals = [particle["coord"][0] for particle in particles]
            y_vals = [particle["coord"][1] for particle in particles]
            ax.scatter(x_vals, y_vals, color=color, s=particle_size, label=f"Processor {i}")
        elif dim == 3:
            x_vals = [particle["coord"][0] for particle in particles]
            y_vals = [particle["coord"][1] for particle in particles]
            z_vals = [particle["coord"][2] for particle in particles]
            ax.scatter(x_vals, y_vals, z_vals, color=color, s=particle_size, label=f"Processor {i}")

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if dim == 3:
        ax.set_zlabel("Z")

    # Add a legend to distinguish processors by color
    #ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

#plot_combined_processors(data_folder)
plot_all_processors_in_subplots(data_folder)
