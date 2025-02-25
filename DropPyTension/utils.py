#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def setup_environment(project_dir, image_filename, csv_filename):
    """
    Set up the working environment, including changing the directory and setting paths.

    Args:
        project_dir (str): Directory for the project.
        image_filename (str): Image file name for the pendant drop.
        csv_filename (str): CSV file name to save results.

    Returns:
        tuple: Contains paths for image and CSV file.
    """
    # Change working directory
    os.chdir(project_dir)

    # Construct file paths
    image_path = os.path.join(project_dir, image_filename)
    csv_file_path = os.path.join(project_dir, csv_filename)

    return image_path, csv_file_path


def setup_constants(density_difference, gravitational_acceleration, needle_diameter):
    """
    Set up constants used in the calculations. The user must provide values for each constant.

    Args:
        density_difference (float): Density difference in kg/m³ (e.g., for fluid in use).
        gravitational_acceleration (float): Gravitational acceleration in m/s² (e.g., 9.80665).
        needle_diameter (float): Needle diameter in meters (e.g., 0.718e-3 m).

    Returns:
        dict: Dictionary of constants used in the analysis.
    
    Raises:
        ValueError: If any of the constants are not provided.
    """
    if density_difference is None:
        raise ValueError("Density difference (density_difference) is required.")
    if gravitational_acceleration is None:
        raise ValueError("Gravitational acceleration (gravitational_acceleration) is required.")
    if needle_diameter is None:
        raise ValueError("Needle diameter (needle_diameter) is required.")

    constants = {
        'delta_rho': density_difference,  # Density difference in kg/m³
        'g': gravitational_acceleration,   # Gravitational acceleration in m/s²
        'needle_diameter': needle_diameter,  # Needle diameter in meters
    }

    return constants

