import os

import gwyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.ndimage import convolve, label
from skimage.morphology import medial_axis, thin
from skimage.measure import regionprops, find_contours
from matplotlib.colors  import ListedColormap


# SETTINGS ==================================================================================================

os.system("cls")

data_dir = Path(r"C:\Users\jorge\OneDrive\Documentos\Documentos\SFSU Master\Python\AFM data")
output_dir = data_dir / "grain_analysis"

# file_name = "Take3_S1_3.5um.gwy"
# file_name = "Take3_S8_3.5um.gwy"

file_name = "S07_A_10x10um_512p_043026.gwy"
# file_name = "S07_A_10x10um_1024p_043026.gwy"
# file_name = "S07_A_10x10um_2048p_043026.gwy"

# file_name = "S003A_2uA_5mM_pH0p85_20250725.gwy"
# file_name = "S003B_10uB_5mM_pH0p85_20250731.gwy"

# file_name = "S56_8-1_2.5um.gwy"
# file_name = "S56_8-1_10um.gwy"

color = "viridis"

channel_name = "Height (retrace)"

threshold_sensitivity_1 = 0.15
threshold_sensitivity_2 = 0.2

min_component_size_1 = 0
min_component_size_2 = 50
min_cycle_size = 10

endpoint_trace_steps = 4
max_complement_steps = 10

medial_axis_rng = 41

overlay_alpha = 0.45
overlay_cmap = "gray"

# KERNELS ==================================================================================================

kernel_laplacian_3 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
])

kernel_laplacian_9 = np.array([
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, -2, -1, 0, 0, 0],
    [0, 0, -1, -2, -3, -2, -1, 0, 0],
    [0, -1, -2, -3, -4, -3, -2, -1, 0],
    [-1, -2, -3, -4, 80, -4, -3, -2, -1],
    [0, -1, -2, -3, -4, -3, -2, -1, 0],
    [0, 0, -1, -2, -3, -2, -1, 0, 0],
    [0, 0, 0, -1, -2, -1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
], dtype=float)

kernel_neighbor = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
], dtype=int)

# FUNCTIONS ==================================================================================================

def automatic_threshold(convolved, direction="negative", sensitivity=3.0):
     
    convolved = np.asarray(convolved, dtype=float)

    median = np.median(convolved)
    mad = np.median(np.abs(convolved - median))

    robust_sigma = 1.4826 * mad

    if robust_sigma == 0:
        robust_sigma = np.std(convolved)

    if robust_sigma == 0:
        thresholded_bool = np.zeros_like(convolved, dtype=bool)
        return (
            thresholded_bool.astype(np.uint8),
            thresholded_bool,
            (median, median),
        )

    lower_threshold = median - sensitivity * robust_sigma
    upper_threshold = median + sensitivity * robust_sigma

    if direction == "negative":
        thresholded_bool = convolved < lower_threshold
    elif direction == "positive":
        thresholded_bool = convolved > upper_threshold
    elif direction == "both":
        thresholded_bool = (
            (convolved < lower_threshold)
            | (convolved > upper_threshold)
        )
    else:
        raise ValueError("direction must be 'negative', 'positive' or 'both'")

    thresholded = thresholded_bool.astype(np.uint8)

    return (
        thresholded,
        thresholded_bool,
        (lower_threshold, upper_threshold),
    )

def remove_small_components(binary_image, min_size=4, connectivity=3):
    
    structure = np.ones((connectivity, connectivity), dtype=int)

    labeled, num_features = label(
        binary_image,
        structure=structure,
    )

    sizes = np.bincount(labeled.ravel())
    filtered = binary_image.copy()

    for group_id in range(1, num_features + 1):
        if sizes[group_id] <= min_size:
            filtered[labeled == group_id] = 0

    return filtered.astype(np.uint8), filtered.astype(bool)

def find_endpoints(skeleton_bool):
    
    neighbor_count = convolve(
        skeleton_bool.astype(np.uint8),
        kernel_neighbor,
        mode="constant",
        cval=0,
    )

    endpoints_bool = skeleton_bool & (neighbor_count == 1)

    return endpoints_bool

def trace_from_endpoint(skeleton_bool, endpoint, max_steps=4):
    
    rows, cols = skeleton_bool.shape

    current = endpoint
    previous = None
    path = [current]

    for _ in range(max_steps):
        r, c = current
        neighbors = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                rr = r + dr
                cc = c + dc

                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue

                if not skeleton_bool[rr, cc]:
                    continue

                neighbor = (rr, cc)

                if neighbor != previous:
                    neighbors.append(neighbor)

        if len(neighbors) == 0:
            break

        if len(neighbors) > 1:
            break

        next_pixel = neighbors[0]

        previous = current
        current = next_pixel
        path.append(current)

    return path

def estimate_endpoint_direction(path):
    
    if len(path) < 2:
        return None

    r_endpoint, c_endpoint = path[0]
    r_inner, c_inner = path[-1]

    dr = r_endpoint - r_inner
    dc = c_endpoint - c_inner

    norm = np.sqrt(dr**2 + dc**2)

    if norm == 0:
        return None

    return np.array([
        dr / norm,
        dc / norm,
    ])

def find_skeleton_continuation_start(skeleton_bool1, skeleton_bool2, endpoint, direction):
    
    if direction is None:
        return []

    rows, cols = skeleton_bool1.shape
    r0, c0 = endpoint

    candidates = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            rr = r0 + dr
            cc = c0 + dc

            if not (0 <= rr < rows and 0 <= cc < cols):
                continue

            if not skeleton_bool1[rr, cc]:
                continue

            if skeleton_bool2[rr, cc]:
                continue

            candidate_vector = np.array([dr, dc], dtype=float)
            candidate_vector /= np.linalg.norm(candidate_vector)

            alignment = np.dot(candidate_vector, direction)

            if alignment > 0:
                candidates.append((rr, cc))

    return candidates

def trace_skeleton_continuation(skeleton_bool1, skeleton_bool2, endpoint, start_pixel, initial_direction, max_steps=5):
    
    rows, cols = skeleton_bool1.shape

    previous = endpoint
    current = start_pixel
    path = [current]

    current_direction = np.array([
        current[0] - previous[0],
        current[1] - previous[1],
    ], dtype=float)

    norm = np.linalg.norm(current_direction)

    if norm > 0:
        current_direction /= norm
    else:
        current_direction = initial_direction.copy()

    for _ in range(max_steps - 1):
        r, c = current
        candidates = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                rr = r + dr
                cc = c + dc

                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue

                neighbor = (rr, cc)

                if neighbor == previous:
                    continue

                if not skeleton_bool1[rr, cc]:
                    continue

                if skeleton_bool2[rr, cc]:
                    continue

                if neighbor in path:
                    continue

                candidate_direction = np.array([dr, dc], dtype=float)
                candidate_direction /= np.linalg.norm(candidate_direction)

                alignment = np.dot(
                    candidate_direction,
                    current_direction,
                )

                candidates.append(
                    (
                        alignment,
                        neighbor,
                        candidate_direction,
                    )
                )

        if len(candidates) == 0:
            break

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_alignment, next_pixel, next_direction = candidates[0]

        if best_alignment <= 0:
            break

        previous = current
        current = next_pixel
        path.append(current)
        current_direction = next_direction

    return path

def complement_skeleton(skeleton_bool1, skeleton_bool2, trace_steps=4, max_complement_steps=5):
    
    endpoints_bool = find_endpoints(skeleton_bool2)
    endpoint_coords = np.argwhere(endpoints_bool)

    endpoint_paths = []
    endpoint_directions = []
    continuation_starts = []
    continuation_paths = []

    for r, c in endpoint_coords:
        path = trace_from_endpoint(
            skeleton_bool2,
            endpoint=(r, c),
            max_steps=trace_steps,
        )

        endpoint_paths.append(path)
        endpoint_directions.append(
            estimate_endpoint_direction(path)
        )

    for (r, c), direction in zip(
        endpoint_coords,
        endpoint_directions,
    ):
        candidates = find_skeleton_continuation_start(
            skeleton_bool1,
            skeleton_bool2,
            endpoint=(r, c),
            direction=direction,
        )

        continuation_starts.append(candidates)

    for endpoint, candidates, direction in zip(
        endpoint_coords,
        continuation_starts,
        endpoint_directions,
    ):
        endpoint_continuations = []

        if direction is not None:
            for start_pixel in candidates:
                path = trace_skeleton_continuation(
                    skeleton_bool1,
                    skeleton_bool2,
                    endpoint=tuple(endpoint),
                    start_pixel=start_pixel,
                    initial_direction=direction,
                    max_steps=max_complement_steps,
                )

                endpoint_continuations.append(path)

        continuation_paths.append(endpoint_continuations)

    skeleton_bool_complemented = skeleton_bool2.copy()

    for endpoint_continuations in continuation_paths:
        for path in endpoint_continuations:
            for r, c in path:
                skeleton_bool_complemented[r, c] = True

    skeleton_bool_complemented = thin(
        skeleton_bool_complemented
    )

    return (
        skeleton_bool_complemented,
        endpoints_bool,
        continuation_paths,
    )

def find_skeleton_bridges(skeleton_bool):
    
    skeleton_bool = np.asarray(
        skeleton_bool,
        dtype=bool,
    )

    pixel_coords = np.argwhere(skeleton_bool)

    pixels = [
        (int(r), int(c))
        for r, c in pixel_coords
    ]

    pixel_set = set(pixels)

    adjacency = {
        pixel: []
        for pixel in pixels
    }

    for r, c in pixels:
        current = (r, c)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                neighbor = (
                    r + dr,
                    c + dc,
                )

                if neighbor in pixel_set:
                    adjacency[current].append(neighbor)

    discovery = {}
    low = {}
    parent = {}
    bridges = []
    time_counter = 0

    for start in pixels:
        if start in discovery:
            continue

        parent[start] = None
        stack = [[start, 0]]

        discovery[start] = time_counter
        low[start] = time_counter
        time_counter += 1

        while stack:
            node, neighbor_index = stack[-1]

            if neighbor_index < len(adjacency[node]):
                neighbor = adjacency[node][neighbor_index]
                stack[-1][1] += 1

                if neighbor not in discovery:
                    parent[neighbor] = node
                    discovery[neighbor] = time_counter
                    low[neighbor] = time_counter
                    time_counter += 1
                    stack.append([neighbor, 0])

                elif neighbor != parent[node]:
                    low[node] = min(
                        low[node],
                        discovery[neighbor],
                    )

            else:
                stack.pop()
                parent_node = parent[node]

                if parent_node is not None:
                    low[parent_node] = min(
                        low[parent_node],
                        low[node],
                    )

                    if low[node] > discovery[parent_node]:
                        bridges.append(
                            (
                                parent_node,
                                node,
                            )
                        )

    return bridges

def keep_only_cycle_structure(skeleton_bool, bridges):
    
    skeleton_bool = np.asarray(
        skeleton_bool,
        dtype=bool,
    )

    bridge_set = {
        tuple(sorted((pixel1, pixel2)))
        for pixel1, pixel2 in bridges
    }

    pixel_coords = np.argwhere(skeleton_bool)

    pixels = [
        (int(r), int(c))
        for r, c in pixel_coords
    ]

    pixel_set = set(pixels)

    cycle_bool = np.zeros_like(
        skeleton_bool,
        dtype=bool,
    )

    for r, c in pixels:
        pixel1 = (r, c)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                pixel2 = (
                    r + dr,
                    c + dc,
                )

                if pixel2 not in pixel_set:
                    continue

                edge = tuple(
                    sorted((pixel1, pixel2))
                )

                if edge not in bridge_set:
                    r1, c1 = pixel1
                    r2, c2 = pixel2

                    cycle_bool[r1, c1] = True
                    cycle_bool[r2, c2] = True

    return cycle_bool

def calculate_surface_area_map(height_nm, pixel_size_x_nm, pixel_size_y_nm):
    
    z = np.asarray(height_nm, dtype=float)

    rows, cols = z.shape

    # Replicate image borders, following the boundary convention
    # used for the surface-area calculation.
    z_pad = np.pad(
        z,
        pad_width=1,
        mode="edge",
    )

    surface_area_pad = np.zeros_like(
        z_pad,
        dtype=float,
    )

    hx = pixel_size_x_nm
    hy = pixel_size_y_nm

    # Each 2x2 group of pixel centres defines one common corner.
    for r in range(rows + 1):
        for c in range(cols + 1):

            z1 = z_pad[r,     c]
            z2 = z_pad[r,     c + 1]
            z3 = z_pad[r + 1, c + 1]
            z4 = z_pad[r + 1, c]

            z_mean = (z1 + z2 + z3 + z4) / 4.0

            # Top triangle: z1 -- z2 -- centre
            A12 = (
                hx * hy / 4.0
                * np.sqrt(
                    1.0
                    + ((z1 - z2) / hx) ** 2
                    + ((z1 + z2 - 2.0 * z_mean) / hy) ** 2
                )
            )

            # Right triangle: z2 -- z3 -- centre
            A23 = (
                hx * hy / 4.0
                * np.sqrt(
                    1.0
                    + ((z2 - z3) / hy) ** 2
                    + ((z2 + z3 - 2.0 * z_mean) / hx) ** 2
                )
            )

            # Bottom triangle: z3 -- z4 -- centre
            A34 = (
                hx * hy / 4.0
                * np.sqrt(
                    1.0
                    + ((z3 - z4) / hx) ** 2
                    + ((z3 + z4 - 2.0 * z_mean) / hy) ** 2
                )
            )

            # Left triangle: z4 -- z1 -- centre
            A41 = (
                hx * hy / 4.0
                * np.sqrt(
                    1.0
                    + ((z4 - z1) / hy) ** 2
                    + ((z4 + z1 - 2.0 * z_mean) / hx) ** 2
                )
            )

            # Half of each triangle belongs to each of the two
            # pixels sharing that triangle.
            surface_area_pad[r, c] += 0.5 * (A12 + A41)

            surface_area_pad[r, c + 1] += 0.5 * (A12 + A23)

            surface_area_pad[r + 1, c + 1] += 0.5 * (A23 + A34)

            surface_area_pad[r + 1, c] += 0.5 * (A34 + A41)

    # Return only the original data-field pixels.
    surface_area_map = surface_area_pad[
        1:rows + 1,
        1:cols + 1,
    ]

    return surface_area_map

def calculate_grain_perimeter(grain_mask, pixel_size_x_nm, pixel_size_y_nm):
    
    contours = find_contours(
        grain_mask.astype(float),
        level=0.5,
    )

    perimeter_nm = 0.0

    for contour in contours:

        if len(contour) < 2:
            continue

        # Close contour if necessary
        contour_closed = np.vstack(
            [
                contour,
                contour[0],
            ]
        )

        dr = np.diff(contour_closed[:, 0])
        dc = np.diff(contour_closed[:, 1])

        distances_nm = np.sqrt(
            (dr * pixel_size_y_nm) ** 2
            + (dc * pixel_size_x_nm) ** 2
        )

        perimeter_nm += np.sum(distances_nm)

    return perimeter_nm

# LOAD AFM DATA ==================================================================================================

file_path = data_dir / file_name
file = gwyfile.load(str(file_path))

data_fields = gwyfile.util.get_datafields(file)
height_field = data_fields[channel_name]

height_m = height_field.data
height_nm = height_m * 1e9

x_size_m = height_field.xreal
y_size_m = height_field.yreal

print()
print("Matrix shape:", height_nm.shape)
print("Physical size:", f"{x_size_m * 1e6:.1f} µm x {y_size_m * 1e6:.1f} µm")
print("Height range:", f"{height_nm.min():.1f} to {height_nm.max():.1f} nm")
print()

# CONVOLUTION ==================================================================================================

convolved1 = convolve(height_nm, kernel_laplacian_3, mode="nearest")

convolved2 = convolve(height_nm, kernel_laplacian_9, mode="nearest")

# AUTOMATIC THRESHOLDING ==================================================================================================

thresholded1, thresholded_bool1, thresholds1 = automatic_threshold(
    convolved1,
    direction="negative",
    sensitivity=threshold_sensitivity_1,
)

thresholded2, thresholded_bool2, thresholds2 = automatic_threshold(
    convolved2,
    direction="negative",
    sensitivity=threshold_sensitivity_2,
)

print(f"Threshold 1: lower={thresholds1[0]:.3f}, ", f"upper={thresholds1[1]:.3f}")
print(f"Threshold 2: lower={thresholds2[0]:.3f}, ", f"upper={thresholds2[1]:.3f}")
print()

# REMOVE SMALL COMPONENTS ==================================================================================================

thresholded_filtered1, thresholded_filtered_bool1 = remove_small_components(thresholded1, min_size=min_component_size_1,)

thresholded_filtered2, thresholded_filtered_bool2 = remove_small_components(thresholded2, min_size=min_component_size_2,)

# SKELETONIZATION ==================================================================================================

# The 3x3 result is used as the more sensitive auxiliary skeleton.
skeleton_bool1 = thin(thresholded_filtered_bool1)

# The 9x9 result is the main skeleton used for the grain boundaries.
# skeleton_bool2 = medial_axis(thresholded_filtered_bool2, rng=medial_axis_rng)
skeleton_bool2 = thin(thresholded_filtered_bool2)

skeleton1 = skeleton_bool1.astype(np.uint8)
skeleton2 = skeleton_bool2.astype(np.uint8)

# COMPLEMENT THE 9x9 SKELETON WITH REAL PIXELS FROM THE 3x3 SKELETON ==================================================================================================

(skeleton_bool_complemented, endpoints_bool, continuation_paths) = complement_skeleton(skeleton_bool1, skeleton_bool2, trace_steps=endpoint_trace_steps, max_complement_steps=max_complement_steps)

skeleton_complemented = (skeleton_bool_complemented.astype(np.uint8))

print("Endpoints in skeleton 2:", np.count_nonzero(endpoints_bool))

added_pixels_bool = (skeleton_bool_complemented & ~skeleton_bool2)

print("Pixels added from skeleton 1:", np.count_nonzero(added_pixels_bool))
print()

# KEEP ONLY CLOSED CYCLE STRUCTURE ==================================================================================================

bridges = find_skeleton_bridges(skeleton_bool_complemented)

skeleton_bool_cycle_cleaned = keep_only_cycle_structure(skeleton_bool_complemented, bridges)

skeleton_cycle_cleaned = (skeleton_bool_cycle_cleaned.astype(np.uint8))

# FINAL SMALL-COMPONENT FILTER ==================================================================================================

skeleton_cycle_filtered, skeleton_cycle_filtered_bool = remove_small_components(skeleton_cycle_cleaned, min_size=min_cycle_size)

print("Bridge connections removed:", len(bridges))
print("Final skeleton pixels:", np.count_nonzero(skeleton_cycle_filtered_bool))
print()

final_mask = skeleton_cycle_filtered.astype(np.uint8)

# GRAIN IDENTIFICATION ==================================================================================================

# Invert the grain mask to identify the grains as connected components
grain_regions = ~final_mask.astype(bool)

labeled_grains, num_regions = label(grain_regions)

# REMOVE GRAINS TOUCHING IMAGE BORDERS ==================================================================================================

border_labels = np.unique(
    np.concatenate(
        [
            labeled_grains[0, :],     # top border
            labeled_grains[-1, :],    # bottom border
            labeled_grains[:, 0],     # left border
            labeled_grains[:, -1],    # right border
        ]
    )
)

border_labels = border_labels[border_labels != 0]

labeled_grains_clean = labeled_grains.copy()
labeled_grains_clean[np.isin(labeled_grains_clean, border_labels)] = 0

valid_grains_mask = labeled_grains_clean > 0

labeled_grains_final, num_valid_grains = label(valid_grains_mask)

print(f"Number of border grains removed: {len(border_labels)}")
print(f"Number of complete grains: {num_valid_grains}")
print()

# STATISTICAL ANALYSIS ========================================================================================

# Physical dimensions of one pixel
rows, cols = height_nm.shape

pixel_size_x_nm = (x_size_m * 1e9) / cols
pixel_size_y_nm = (y_size_m * 1e9) / rows

pixel_area_nm2 = (
    pixel_size_x_nm
    * pixel_size_y_nm
)

print("Pixel size:", f"{pixel_size_x_nm:.3f} nm x ", f"{pixel_size_y_nm:.3f} nm")
print("Projected pixel area:", f"{pixel_area_nm2:.3f} nm²")
print()

# SURFACE AREA MAP --------------------------------------------------------------------------------------------------

surface_area_map = calculate_surface_area_map(height_nm, pixel_size_x_nm, pixel_size_y_nm)

# REGION PROPERTIES --------------------------------------------------------------------------------------------------

grain_properties = regionprops(labeled_grains_final, spacing=(pixel_size_y_nm,pixel_size_x_nm))

# ANALYZE EACH GRAIN --------------------------------------------------------------------------------------------------

grain_statistics = []

for region in grain_properties:

    grain_id = region.label
    grain_mask = (labeled_grains_final == grain_id)

    # BASIC SIZE ----------------------------------------------------------------------------------------------

    pixel_count = np.count_nonzero(grain_mask)

    projected_area_nm2 = (pixel_count * pixel_area_nm2)

    equivalent_diameter_nm = (2.0 * np.sqrt(projected_area_nm2 / np.pi))

    # PERIMETER AND SHAPE ----------------------------------------------------------------------------------------------

    perimeter_nm = calculate_grain_perimeter(grain_mask, pixel_size_x_nm, pixel_size_y_nm)

    if perimeter_nm > 0:

        circularity = (4.0 * np.pi * projected_area_nm2 / perimeter_nm**2)

    else:
        circularity = np.nan

    major_axis_nm = (region.axis_major_length)

    minor_axis_nm = (region.axis_minor_length)

    if minor_axis_nm > 0:

        aspect_ratio = (major_axis_nm / minor_axis_nm)

    else:
        aspect_ratio = np.nan

    # regionprops defines orientation relative to the row axis. Convert it to an angle relative to the horizontal x-axis.

    orientation_deg = (90.0 - np.degrees(region.orientation))

    # Normalize to [-90, 90)

    orientation_deg = ((orientation_deg + 90.0) % 180.0 - 90.0)


    # HEIGHT STATISTICS ----------------------------------------------------------------------------------------------

    grain_heights_nm = (height_nm[grain_mask])

    min_height_nm = np.min(grain_heights_nm)

    max_height_nm = np.max(grain_heights_nm)

    mean_height_nm = np.mean(grain_heights_nm)

    median_height_nm = np.median(grain_heights_nm)

    rms_height_nm = np.sqrt(np.mean((grain_heights_nm - mean_height_nm) ** 2))

    height_range_nm = (max_height_nm - min_height_nm)

    # SURFACE AREA ----------------------------------------------------------------------------------------------

    surface_area_nm2 = np.sum(surface_area_map[grain_mask])

    # Developed surface-area ratio: Percentage of additional area produced by topography.

    if projected_area_nm2 > 0:

        sdr_percent = ((surface_area_nm2 - projected_area_nm2) / projected_area_nm2 * 100.0)

    else:
        sdr_percent = np.nan

    # STORE RESULTS ----------------------------------------------------------------------------------------------

    grain_statistics.append(
        {
            "Grain_ID": grain_id,

            "Pixel_count": pixel_count,

            "Projected_area_nm2":
                projected_area_nm2,

            "Equivalent_diameter_nm":
                equivalent_diameter_nm,

            "Perimeter_nm":
                perimeter_nm,

            "Circularity":
                circularity,

            "Major_axis_nm":
                major_axis_nm,

            "Minor_axis_nm":
                minor_axis_nm,

            "Aspect_ratio":
                aspect_ratio,

            "Orientation_deg":
                orientation_deg,

            "Min_height_nm":
                min_height_nm,

            "Max_height_nm":
                max_height_nm,

            "Mean_height_nm":
                mean_height_nm,

            "Median_height_nm":
                median_height_nm,

            "Height_range_nm":
                height_range_nm,

            "RMS_height_nm":
                rms_height_nm,

            "Surface_area_nm2":
                surface_area_nm2,

            "Sdr_percent":
                sdr_percent,
        }
    )

# CREATE DATAFRAME --------------------------------------------------------------------------------------------------

grain_statistics_df = pd.DataFrame(grain_statistics)

# EXPORT CSV --------------------------------------------------------------------------------------------------

output_dir.mkdir(parents=True, exist_ok=True,)

output_file = (output_dir / f"{Path(file_name).stem}_grain_statistics.csv")

grain_statistics_df.to_csv(output_file, index=False)

print(f"Grain statistics exported to:\n", f"{output_file}")
print(f"Total grains exported: ", f"{len(grain_statistics_df)}")
print()

# PLOT SETTINGS ================================================================================================

afm_cmap = "viridis"

boundary_color = "black"
boundary_alpha = 0.90

valid_grain_color = "black"
valid_grain_alpha = 0.25


output_dir.mkdir(parents=True, exist_ok=True)
plot_dpi = 600

# PLOT 1 =======================================================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

axes[0].imshow(height_nm, cmap=afm_cmap, origin="upper")
axes[0].set_title("AFM")
axes[0].axis("off")

axes[1].imshow(height_nm, cmap=afm_cmap, origin="upper")
axes[1].imshow(np.ma.masked_where(~final_mask.astype(bool), final_mask), cmap=ListedColormap([boundary_color]), alpha=boundary_alpha, origin="upper")
axes[1].set_title("Detected grain boundaries")
axes[1].axis("off")

plt.tight_layout()

plot1_file = (
    output_dir
    / f"{Path(file_name).stem}_AFM_and_boundaries.png"
)

fig.savefig(
    plot1_file,
    dpi=plot_dpi,
    bbox_inches="tight",
)

print(f"Plot 1 exported to:\n{plot1_file}")

plt.show()

# PLOT 2 =======================================================================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(height_nm, cmap=afm_cmap, origin="upper")

ax.imshow(
    np.ma.masked_where(
        ~valid_grains_mask,
        valid_grains_mask,
    ),
    cmap=ListedColormap(
        [valid_grain_color]
    ),
    alpha=valid_grain_alpha,
    origin="upper",
)

ax.set_title(f"Valid grains over AFM\n Grains: {num_valid_grains}")
ax.axis("off")

plt.tight_layout()

plot2_file = (
    output_dir
    / f"{Path(file_name).stem}_valid_grains_overlay.png"
)

fig.savefig(
    plot2_file,
    dpi=plot_dpi,
    bbox_inches="tight",
)

print(f"Plot 2 exported to:\n{plot2_file}")

plt.show()

# PLOT 3 =======================================================================================================

# 0 = background
# 1 = valid grain

binary_export_mask = (valid_grains_mask.astype(np.uint8) * 255)

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(binary_export_mask, cmap="gray", vmin=0, vmax=255, origin="upper",)
ax.set_title(f"Binary mask\n Grains: {num_valid_grains}")
ax.axis("off")

plt.tight_layout()

# Export complete Plot 3
plot3_file = (
    output_dir
    / f"{Path(file_name).stem}_binary_mask_plot.png"
)

fig.savefig(
    plot3_file,
    dpi=plot_dpi,
    bbox_inches="tight",
)

print(f"Plot 3 exported to:\n{plot3_file}")


# Export raw binary mask

mask_output_file = (
    output_dir
    / f"{Path(file_name).stem}_grain_mask.png"
)

plt.imsave(
    mask_output_file,
    binary_export_mask,
    cmap="gray",
    vmin=0,
    vmax=255,
)

print(f"Binary grain mask exported to:\n{mask_output_file}")

plt.show()
