import os
import tempfile
import gwyfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from scipy.ndimage import convolve, label
from plotly.subplots import make_subplots
from skimage.morphology import thin, medial_axis
from skimage.measure import regionprops, find_contours
from io import BytesIO

# KERNELS =====================================================================================================

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

# APP ==================================================================================================

st.title("AFM Grain Analysis")

st.write("Web interface for AFM grain detection and statistical analysis.")

# UPLOAD AFM FILE =============================================================================================

uploaded_file = st.file_uploader("Upload an AFM file", type=["gwy"])

# LOAD AFM DATA ================================================================================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gwy",) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:

        file = gwyfile.load(temp_path)

        data_fields = gwyfile.util.get_datafields(file)

        # AVAILABLE CHANNELS ----------------------------------------------------------------------------------

        channel_names = list(data_fields.keys())

        selected_channel = st.selectbox("Select AFM channel", channel_names)

        # LOAD SELECTED CHANNEL ----------------------------------------------------------------------------------

        height_field = data_fields[selected_channel]

        height_m = height_field.data
        height_nm = height_m * 1e9

        x_size_m = height_field.xreal
        y_size_m = height_field.yreal

        # DISPLAY AFM IMAGE ----------------------------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(6, 6))

        image = ax.imshow(height_nm, cmap="viridis", origin="upper")

        ax.set_title(selected_channel)
        ax.axis("off")

        fig.colorbar(image, ax=ax, label="Height (nm)", fraction=0.046, pad=0.04)

        st.pyplot(fig)

        plt.close(fig)

        # DISPLAY BASIC INFORMATION ----------------------------------------------------------------------------------

        st.write("Matrix shape:", height_nm.shape)
        st.write("Physical size:", f"{x_size_m * 1e6:.2f} µm × {y_size_m * 1e6:.2f} µm")
        st.write("Height range:", f"{height_nm.min():.2f} to {height_nm.max():.2f} nm")

        # SETTINGS ----------------------------------------------------------------------------------

        st.subheader("Detection settings")

        threshold_sensitivity_1 = st.number_input(
            "3×3 threshold sensitivity",
            min_value=0.0,
            value=0.15,
            step=0.01,
        )

        threshold_sensitivity_2 = st.number_input(
            "9×9 threshold sensitivity",
            min_value=0.0,
            value=0.20,
            step=0.01,
        )

        min_component_size_1 = st.number_input(
            "3×3 minimum component size",
            min_value=0,
            value=0,
            step=1,
        )

        min_component_size_2 = st.number_input(
            "9×9 minimum component size",
            min_value=0,
            value=50,
            step=1,
        )

        endpoint_trace_steps = st.number_input(
            "Endpoint trace steps",
            min_value=1,
            value=4,
            step=1,
        )

        max_complement_steps = st.number_input(
            "Maximum complement steps",
            min_value=1,
            value=10,
            step=1,
        )

        min_cycle_size = st.number_input(
            "Minimum cycle size",
            min_value=0,
            value=10,
            step=1,
        )

        # CONVOLUTION ----------------------------------------------------------------------------------

        convolved1 = convolve(height_nm, kernel_laplacian_3, mode="nearest")

        convolved2 = convolve(height_nm, kernel_laplacian_9, mode="nearest")

        # AUTOMATIC THRESHOLDING ----------------------------------------------------------------------------------

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

        # REMOVE SMALL COMPONENTS ----------------------------------------------------------------------------------

        thresholded_filtered1, thresholded_filtered_bool1 = remove_small_components(
            thresholded1,
            min_size=min_component_size_1,
        )

        thresholded_filtered2, thresholded_filtered_bool2 = remove_small_components(
            thresholded2,
            min_size=min_component_size_2,
        )

        # SKELETONIZATION ----------------------------------------------------------------------------------

        skeleton_bool1 = thin(thresholded_filtered_bool1)
        skeleton_bool2 = thin(thresholded_filtered_bool2)

        skeleton1 = skeleton_bool1.astype(np.uint8)
        skeleton2 = skeleton_bool2.astype(np.uint8)

        # ENDPOINT DETECTION ----------------------------------------------------------------------------------

        endpoints_bool = find_endpoints(skeleton_bool2)
        endpoint_coords = np.argwhere(endpoints_bool)
        num_endpoints = np.count_nonzero(endpoints_bool)

        # ESTIMATE ENDPOINT DIRECTIONS ----------------------------------------------------------------------------------

        endpoint_paths = []
        endpoint_directions = []

        for r, c in endpoint_coords:

            path = trace_from_endpoint(skeleton_bool2, endpoint=(r, c), max_steps=endpoint_trace_steps)

            direction = estimate_endpoint_direction(path)

            endpoint_paths.append(path)

            endpoint_directions.append(direction)

        # FIND CONTINUATION STARTS ----------------------------------------------------------------------------------

        continuation_starts = []

        for (r, c), direction in zip(endpoint_coords, endpoint_directions):

            candidates = find_skeleton_continuation_start(
                skeleton_bool1,
                skeleton_bool2,
                endpoint=(r, c),
                direction=direction,
            )

            continuation_starts.append(candidates)

        # TRACE SKELETON CONTINUATIONS ----------------------------------------------------------------------------------

        continuation_paths = []

        for endpoint, candidates, direction in zip(endpoint_coords, continuation_starts, endpoint_directions):

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

        # COMPLEMENT 9x9 SKELETON ----------------------------------------------------------------------------------

        skeleton_bool_complemented = skeleton_bool2.copy()

        for endpoint_continuations in continuation_paths:
            for path in endpoint_continuations:
                for r, c in path:
                    skeleton_bool_complemented[r, c] = True

        skeleton_bool_complemented = thin(skeleton_bool_complemented)

        skeleton_complemented = (skeleton_bool_complemented.astype(np.uint8))

        added_pixels_bool = (skeleton_bool_complemented & ~skeleton_bool2)

        num_added_pixels = np.count_nonzero(added_pixels_bool)

        # KEEP ONLY CLOSED CYCLE STRUCTURE ---------------------------------------------------------------------------

        bridges = find_skeleton_bridges(skeleton_bool_complemented)

        skeleton_bool_cycle_cleaned = keep_only_cycle_structure(skeleton_bool_complemented, bridges)

        skeleton_cycle_cleaned = (skeleton_bool_cycle_cleaned.astype(np.uint8))

        # FINAL SMALL-COMPONENT FILTER -------------------------------------------------------------------------------

        skeleton_cycle_filtered, skeleton_cycle_filtered_bool = remove_small_components(skeleton_bool_cycle_cleaned.astype(np.uint8), min_size=min_cycle_size)

        final_mask = skeleton_cycle_filtered.astype(np.uint8)

        # GRAIN IDENTIFICATION ----------------------------------------------------------------------------------

        grain_regions = ~final_mask.astype(bool)

        labeled_grains, num_regions = label(grain_regions)

        # REMOVE GRAINS TOUCHING IMAGE BORDERS ------------------------------------------------------------------

        border_labels = np.unique(
            np.concatenate(
                [
                    labeled_grains[0, :],
                    labeled_grains[-1, :],
                    labeled_grains[:, 0],
                    labeled_grains[:, -1],
                ]
            )
        )

        border_labels = border_labels[border_labels != 0]

        labeled_grains_clean = labeled_grains.copy()

        labeled_grains_clean[np.isin(labeled_grains_clean,border_labels)] = 0

        valid_grains_mask = (labeled_grains_clean > 0)

        labeled_grains_final, num_valid_grains = label(valid_grains_mask)

        # CREATE BINARY GRAIN MASK ----------------------------------------------------------------------------------

        binary_export_mask = (valid_grains_mask.astype(np.uint8) * 255)

        # PIXEL PHYSICAL DIMENSIONS ----------------------------------------------------------------------------------

        rows, cols = height_nm.shape

        pixel_size_x_nm = (x_size_m * 1e9) / cols
        pixel_size_y_nm = (y_size_m * 1e9) / rows
        pixel_area_nm2 = (pixel_size_x_nm * pixel_size_y_nm)

        # SURFACE AREA MAP ----------------------------------------------------------------------------------

        surface_area_map = calculate_surface_area_map(height_nm, pixel_size_x_nm, pixel_size_y_nm)

        # REGION PROPERTIES ----------------------------------------------------------------------------------

        grain_properties = regionprops(labeled_grains_final, spacing=(pixel_size_y_nm, pixel_size_x_nm))

        # ANALYZE EACH GRAIN ----------------------------------------------------------------------------------

        grain_statistics = []

        for region in grain_properties:

            grain_id = region.label
            grain_mask = (labeled_grains_final == grain_id)

            # BASIC SIZE --------------------------------------------------------------------------------------

            pixel_count = np.count_nonzero(grain_mask)
            projected_area_nm2 = (pixel_count * pixel_area_nm2)
            equivalent_diameter_nm = (2.0 * np.sqrt(projected_area_nm2 / np.pi))

            # PERIMETER AND SHAPE -----------------------------------------------------------------------------

            perimeter_nm = calculate_grain_perimeter(grain_mask, pixel_size_x_nm, pixel_size_y_nm)

            if perimeter_nm > 0:

                circularity = (4.0 * np.pi * projected_area_nm2 / perimeter_nm**2)

            else:

                circularity = np.nan

            # AXIS AND ORIENTATION -----------------------------------------------------------------------------

            major_axis_nm = (region.axis_major_length)
            minor_axis_nm = (region.axis_minor_length)

            if minor_axis_nm > 0:

                aspect_ratio = (major_axis_nm / minor_axis_nm)

            else:

                aspect_ratio = np.nan

            orientation_deg = (90.0 - np.degrees(region.orientation))
            orientation_deg = ((orientation_deg + 90.0) % 180.0 - 90.0)

            # HEIGHT STATISTICS -------------------------------------------------------------------------------

            grain_heights_nm = (height_nm[grain_mask])
            min_height_nm = np.min(grain_heights_nm)
            max_height_nm = np.max(grain_heights_nm)
            mean_height_nm = np.mean(grain_heights_nm)
            median_height_nm = np.median(grain_heights_nm)

            rms_height_nm = np.sqrt(np.mean((grain_heights_nm - mean_height_nm) ** 2))

            height_range_nm = (max_height_nm - min_height_nm)

            # SURFACE AREA -------------------------------------------------------------------------------------

            surface_area_nm2 = np.sum(surface_area_map[grain_mask])

            if projected_area_nm2 > 0:

                sdr_percent = ((surface_area_nm2 - projected_area_nm2) / projected_area_nm2 * 100.0)

            else:

                sdr_percent = np.nan

            # STORE RESULTS ------------------------------------------------------------------------------------

            grain_statistics.append(
                {
                    "Grain_ID":
                        grain_id,

                    "Pixel_count":
                        pixel_count,

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
        # CREATE DATAFRAME ----------------------------------------------------------------------------------

        grain_statistics_df = pd.DataFrame(grain_statistics)

        # DISPLAY STATISTICAL RESULTS ----------------------------------------------------------------------------------

        st.subheader("Grain statistics table")
        st.write(f"Total grains analyzed: {len(grain_statistics_df)}")
        st.dataframe(grain_statistics_df,use_container_width=True,)

        # PREPARE CSV DOWNLOAD ----------------------------------------------------------------------------------

        csv_data = grain_statistics_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download grain statistics CSV",
            data=csv_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_grain_statistics.csv",
            mime="text/csv",
        )

        # PREPARE BINARY MASK DOWNLOAD -----------------------------------------------------------------------

        mask_buffer = BytesIO()

        plt.imsave(
            mask_buffer,
            binary_export_mask,
            cmap="gray",
            vmin=0,
            vmax=255,
            format="png",
        )

        mask_buffer.seek(0)

        st.download_button(
            label="Download grain mask PNG",
            data=mask_buffer,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_grain_mask.png",
            mime="image/png",
        )

        # DISPLAY FILTERING RESULTS ----------------------------------------------------------------------------------

        st.subheader("Filtering results")

        fig_filter, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(thresholded_bool1, cmap="gray_r", origin="upper")
        axes[0, 0].set_title(f"3×3 threshold mask\nThreshold = {thresholds1[0]:.3f}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(thresholded_bool2, cmap="gray_r", origin="upper")
        axes[0, 1].set_title(f"9×9 threshold mask\nThreshold = {thresholds2[0]:.3f}")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(thresholded_filtered_bool1, cmap="gray_r", origin="upper")
        axes[1, 0].set_title(f"3×3 filtered mask\nMin size = {min_component_size_1}")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(thresholded_filtered_bool2, cmap="gray_r", origin="upper")
        axes[1, 1].set_title(f"9×9 filtered mask\nMin size = {min_component_size_2}")
        axes[1, 1].axis("off")

        plt.tight_layout()
        st.pyplot(fig_filter)
        plt.close(fig_filter)

        # DISPLAY SKELETON RESULTS ----------------------------------------------------------------------------------

        st.subheader("Skeletonization results")

        fig_skeleton, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        axes[0].imshow(skeleton_bool1, cmap="gray_r", origin="upper")
        axes[0].set_title("3×3 skeleton")
        axes[0].axis("off")

        axes[1].imshow(skeleton_bool2, cmap="gray_r", origin="upper")
        axes[1].set_title("9×9 skeleton")
        axes[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig_skeleton)
        plt.close(fig_skeleton)

        # DISPLAY ENDPOINTS ----------------------------------------------------------------------------------

        st.subheader("Endpoint detection")

        fig_endpoints, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(skeleton_bool2, cmap="gray_r",origin="upper")

        if len(endpoint_coords) > 0:

            ax.scatter(endpoint_coords[:, 1], endpoint_coords[:, 0], s=12, c="red")

        ax.set_title(f"9×9 skeleton endpoints\nEndpoints = {num_endpoints}")
        ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig_endpoints)
        plt.close(fig_endpoints)

        # DISPLAY COMPLEMENTATION ----------------------------------------------------------------------------------

        st.subheader("Skeleton complementation")

        st.write(f"Pixels added from 3×3 skeleton: {num_added_pixels}")

        fig_complemented, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        axes[0].imshow(skeleton_bool2, cmap="gray_r", origin="upper")
        axes[0].set_title("Original 9×9 skeleton")
        axes[0].axis("off")

        axes[1].imshow(skeleton_bool_complemented, cmap="gray_r", origin="upper")
        axes[1].set_title(f"Complemented skeleton\nAdded pixels = {num_added_pixels}")
        axes[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig_complemented)
        plt.close(fig_complemented)

        # DISPLAY CYCLE STRUCTURE ----------------------------------------------------------------------------------
        
        st.subheader("Cycle structure")

        fig_cycle, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        axes[0].imshow(skeleton_bool_complemented, cmap="gray_r", origin="upper")
        axes[0].set_title("Complemented skeleton")
        axes[0].axis("off")

        axes[1].imshow(skeleton_bool_cycle_cleaned, cmap="gray_r", origin="upper")
        axes[1].set_title(f"Closed-cycle structure\nBridges removed = {len(bridges)}")
        axes[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig_cycle)
        plt.close(fig_cycle)

        # DISPLAY FINAL MASK ----------------------------------------------------------------------------------

        st.subheader("Final grain-boundary mask")

        fig_final_mask, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(final_mask, cmap="gray_r", origin="upper")
        ax.set_title(f"Final boundary mask\nMinimum cycle size = {min_cycle_size}")
        ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig_final_mask)
        plt.close(fig_final_mask)

        # DISPLAY VALID GRAINS ----------------------------------------------------------------------------------

        st.subheader("Detected grains")
        st.write(f"Number of complete grains: {num_valid_grains}")
        st.write(f"Grains removed at image borders: {len(border_labels)}")

        fig_grains, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(height_nm, cmap="viridis", origin="upper")
        ax.imshow(np.ma.masked_where(~valid_grains_mask, valid_grains_mask,), cmap="gray", alpha=0.30, origin="upper")
        ax.set_title(f"Detected complete grains\nGrains = {num_valid_grains}")
        ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig_grains)
        plt.close(fig_grains)



    finally:

        os.remove(temp_path)