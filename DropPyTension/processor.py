#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scipy.optimize import leastsq
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from scipy.interpolate import interp1d

import csv
from collections import defaultdict
from datetime import datetime


class ImageProcessor:
    def __init__(self, image_path):
        """
        Initialize the ImageProcessor class.
        
        Args:
            image_path (str): Path to the image file.
        """
        # Try to load the image
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"The file at {image_path} is not a valid image or cannot be loaded.")

        self.roi = None  # Placeholder for ROI coordinates (x1, z1, x2, z2)
        self.roi_image = None  # Placeholder for ROI Image
        self.gray_image = None # Placeholder for the grayscale image
        self.blurry_image = None # Placeholder for the blurred image
        self.binary_mask = None # Placeholder for the binary mask
        self.edges = None# Placeholder for image edges
        self.drop_points = None # Placeholder for drop contour points
        self.tip_position = None # Placeholder for tip position
        self.tip_pixels = None # Placeholder for the tip pixels
        self.tip_width = None # Placeholder for the tip width
        self.origin_shift = None # Placeholder for the origin shift
        self.drop_points_cartesian = None  # Placeholder for drop contour points in cart coordinates
        self.points_cartesian = None # Placeholder for point coordinates in m
        self.points_pixels = None # Placeholder for point coordinates in pixel
        self.drop_volume = None # Placeholder for drop volume
        self.interfacial_values = None # Placeholder for interfacial tension values
        
    def print_attributes(self):
        # Print instance attributes using vars() or __dict__
        for attribute, value in vars(self).items():
            print(f'{attribute}: {value}')
        
    
    def draw_image(self, img=None):
        """
        Draw a given image from the processor. If no image is provided, 
        it will attempt to use self.image.
    
        Args:
            image (numpy.ndarray, optional): The image to display. Defaults to None, using self.image.
        """
        # Use the provided image or fall back to self.image
        if img is None:
            img = self.image
    
        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
            plt.title("Image Display")
            plt.axis('off')  # Hide axes
            plt.show()
        else:
            print("No image available to display. Please provide a valid image.")


    def set_roi(self, x1, z1, x2, z2):
        """
        Set the Region of Interest (ROI) on the image.
        
        Args:
            x1, z1 (int): Coordinates for the top-left corner of the ROI.
            x2, z2 (int): Coordinates for the bottom-right corner of the ROI.
        """
        self.roi = (x1, z1, x2, z2)
        return self.roi

    def draw_roi(self):
        """
        Draw the ROI on the image copy and display it.
        """
        if self.roi is not None:
            x1, z1, x2, z2 = self.roi
            img = self.image.copy()
            # Draw a rectangle on the image
            cv2.rectangle(img, (x1, z1), (x2, z2), (0, 255, 0), 2)
            # Display the image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
            plt.title("Original Image with ROI")
            plt.show()
        else:
            print("ROI is not set. Please use set_roi() to define the ROI.")
        return


    def extract_roi_image(self):
        """
        Extract the Region of Interest (ROI) from the image based on the defined coordinates.
        """
        if self.roi is not None:
            x1, z1, x2, z2 = self.roi
            self.roi_image = self.image[z1:z2, x1:x2]
            return self.roi_image
        else:
            print("ROI is not set. Please use set_roi() to define the ROI.")
            return
    
    def make_gray_image(self, img=None):
        """
        Convert the image or ROI image to grayscale.
        
        Args:
            roi_image (numpy.ndarray): The image to process. Defaults to None.
        
        Returns:
            numpy.ndarray: Grayscale image.
        """
        if img is None:
            raise ValueError("Image is None. Please provide a valid image.")
        
        # Convert the image to grayscale
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def make_blurry_image(self, img=None, kernel_size=(5, 5), sigma=3):
        """
        Apply Gaussian blur to the grayscale image.
        
        Args:
            kernel_size (tuple): The size of the kernel to be used for Gaussian blur.
            sigma (int): The standard deviation used in the Gaussian kernel.
            roi_image (numpy.ndarray): The image to process
        
        Returns:
            numpy.ndarray: Blurred image.
        """
        if img is None:
            raise ValueError("Image is None. Please provide a valid image.")
        
        # Apply Gaussian blur to the image
        self.blurry_image = cv2.GaussianBlur(self.gray_image, kernel_size, sigma)
        return self.blurry_image
    
    
    def apply_binary_threshold(self, img=None, threshold_value=127, max_value=255):
        """
        Apply binary thresholding to the grayscale or blurred image.
        
        Args:
            threshold_value (int): The threshold value to convert to binary.
            max_value (int): The maximum value for the binary mask.
            roi_image (numpy.ndarray): The image to process. Defaults to None, using self.image.
        
        Returns:
            numpy.ndarray: Binary thresholded image.
        """
        if img is None:
            raise ValueError("Image is None. Please provide a valid image.")

        
        # Apply binary thresholding
        _, self.binary_mask = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)
        return self.binary_mask
    
    
    def process_roi_image(self):
        """
        Streamlined function to convert the image to grayscale, apply blurring, 
        and then apply a binary threshold to the image.
    
        Returns:
            np.array: The binary thresholded image.
        """
        # Step 1: Make the image grayscale using the region of interest image (ROI)
        self.make_gray_image(img=self.roi_image)
        self.draw_image(img=self.gray_image)
        
        # Step 2: Apply blurring to the grayscale image
        self.make_blurry_image(img=self.gray_image)
        self.draw_image(img=self.blurry_image)
        
        # Step 3: Apply binary thresholding to the blurred image
        self.apply_binary_threshold(img=self.blurry_image)
        return
    

    def extract_tip_pixels(self, tip_position):
        """
        Extract the coordinates of the black pixels above a given tip position in an image.
        
        Returns:
            list of tuples: A list of (x, y) coordinates of the black pixels above the tip.
        """
        # Use the ROI image if provided, else use self.image
        if self.binary_mask is None:
            raise ValueError("binary_mask is None. Please use apply_binary_threshold on image.")

        # Initialize a list to store the coordinates of black pixels
        tip_pixel_coordinates = []

        # Iterate over the rows above the tip position
        for y in range(tip_position):
            for x in range(self.binary_mask.shape[1]):
                # If the pixel is black (0 in the binary mask), save its coordinates
                if self.binary_mask[y, x] <= 200:  # Check for black pixels 
                    tip_pixel_coordinates.append((x, y))
        self.tip_pixels = tip_pixel_coordinates
        self.tip_position = tip_position
        return self.tip_pixels


    def draw_tip(self):
        """
        Draw the extracted tip pixel coordinates on the figure using a mask.
        """
        if self.tip_pixels is None:
            raise ValueError("Missing tip pixel coordinates. Please use extract_tip_pixels first.")

        # Create a mask with the same dimensions as the roi_image
        mask = np.zeros_like(self.roi_image)
    
        # Set the pixels in the mask for the extracted tip coordinates
        for x, y in self.tip_pixels:
            mask[y, x] = [255, 0, 0]  # Set to red color (BGR format)
            
        # Draw the results on the figure
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Combine the mask with the ROI image
        combined_image = cv2.addWeighted(self.roi_image, 1, mask, 0.4, 0)
    
        # Display the combined image
        ax.imshow(combined_image)  
    
        ax.axhline(self.tip_position, linestyle=':', linewidth=1, color='red', alpha=0.5)

        ax.set_title('Tip Pixel Coordinates Highlighted')
        ax.set_xlabel('X-axis (Pixels)')
        ax.set_ylabel('Z-axis (Pixels)')
    
        return fig, ax

    
    def get_image_scale(self, tip_diameter):
        """
        Wrapper method to calculate the average width from the extracted tip pixels
        and compute the scale from the tip width and known tip diameter.
        
        Returns:
            tuple: (tip_width, error), 
            tuple: (scale, error)
        """
        # Check if tip_pixels is available
        if self.tip_pixels is None:
            raise ValueError("tip_pixels is None. Please extract tip pixels first.")
        
        # Calculate the width from the tip pixels
        self.tip_width = calculate_width(self.tip_pixels)
        
        # Calculate the scale using the tip width and the known tip diameter
        self.scale = get_scale(self.tip_width, tip_diameter)
        
        # Return the calculated values (average width and scale)
        return self.tip_width, self.scale
    
    
    def find_drop_contour(self, canny_thresholds = {'threshold1': 50, 'threshold2': 100}, smoothing = None ):
        
        if self.tip_position is None:
            raise ValueError("tip_position is None. Please use extract_tip_pixels.")
        
        img = self.blurry_image
        
        # Crop below the tip (pixels above in image coordinates)
        cropped_img = img[self.tip_position:, :]
    
        # Perform Canny edge detection
        self.edges = cv2.Canny(cropped_img, canny_thresholds['threshold1'], canny_thresholds['threshold2'])  # Adjust thresholds as necessary
    
        # Display the edges
        plt.imshow(self.edges, cmap='gray')
        plt.title("Detected Edges in ROI")
        plt.axis('off')
        plt.show()
        
        # Find contours in the edges
        self.drop_points, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not self.drop_points:
            raise ValueError("No contours found. Adjust Canny parameters or check input.")
    
    
        # Find the lowest contour (outermost bottom one)
        outermost_contour = max(self.drop_points, key=lambda c: c[:, :, 1].max())  
    
        # Extract contour points
        self.drop_points = outermost_contour[:, 0, :]
    
        # Adjust y-coordinates to match original image
        self.drop_points[:, 1] += self.tip_position  
        
        # Orders points in a clockwise manner and moves the point closest to z=0 to the beginning
        self.drop_points = order_points(self.drop_points)
        
        if smoothing is not None:
            self.drop_points = smooth_savgol(self.drop_points, window_length=smoothing['window_length'], polyorder=smoothing['polyorder'])
        return

    def calibrate(self, proportion=0.15):
        """
        Calibrate the drop contour by fitting a circle to the bottom points of the drop contour,
        adjusting the origin to the bottom of the fitted circle, and plotting the results.
    
        Args:
            proportion (float, optional): Proportion of points from the bottom to fit the circle.
                                          Default is 0.15 (15% of the drop contour).
    
        Returns:
            tuple: A tuple containing the `fig` and `ax` objects for the generated plot.
                  This method also modifies the `origin_shift` and `drop_points_cartesian` attributes of the object.
        """
    
        # Step 1: Fit a circle to the bottom contour points. The `fit_circle` function is used,
        #         where `np.argmax(self.drop_points[:, 1])` finds the index of the highest point (tip) in the contour.
        #         The proportion of points from the bottom is used to determine which points to fit the circle on.
        x0, z0, r0 = fit_circle(self.drop_points, np.argmax(self.drop_points[:, 1]), proportion)
    
        # Step 2: Adjust the origin to the bottom of the fitted circle by shifting the center (x0, z0) downwards
        #         by the radius of the fitted circle. This defines the new origin of the drop in the coordinate system.
        self.origin_shift = np.array([x0, z0 + r0])  # Store the new origin point at the bottom of the drop
    
        # Step 3: Convert the contour points to Cartesian coordinates (in millimeters).
        #         The conversion involves subtracting the origin shift from the points and scaling them.
        self.drop_points_cartesian = (self.origin_shift - self.drop_points) / self.scale[0]  # Convert X to mm
    
        # Step 4: Create a plot to visualize the fitted circle and origin point on the image.
        #         The plot includes the original region of interest (ROI) image and the fitted circle.
        fig, ax = plt.subplots()  # Create a figure and axis for plotting
    
        plt.imshow(self.roi_image, cmap='gray')  # Display the selected region of interest (ROI) image
    
        # Plot the fitted circle using the center (x0, z0) and radius (r0) with a dashed green line
        circle_plot = plt.Circle((x0, z0), r0, color='green', fill=False, linestyle='--', linewidth=2, label="Fitted Circle")
        plt.gca().add_patch(circle_plot)  # Add the circle patch to the plot
    
        # Mark the new origin with a red dot at the bottom of the drop
        plt.scatter(self.origin_shift[0], self.origin_shift[1], c='red', s=25, label='Origin')
    
        # Step 5: Customize the plot with title, labels, and legend
        plt.title("Fitted Circle and Origin point on ROI Image")  # Title of the plot
        plt.xlabel("X (pixels)")  # X-axis label
        plt.ylabel("Y (pixels)")  # Y-axis label
        plt.legend(loc="upper right")  # Display the legend
    
        # Return the figure and axis objects for further manipulation or saving if needed
        return fig, ax
    
    def extract_points_of_interest(self):
        self.points_cartesian = {}
        square_dict = smallest_bounding_square(self.drop_points_cartesian)
        self.points_cartesian.update(square_dict)
        
        ds_dict = get_intersections_at_z(points = self.drop_points_cartesian, z = square_dict['top_left'][1])
        self.points_cartesian['Ds_right'] = ds_dict['right']
        self.points_cartesian['Ds_left']  = ds_dict['left']
        
        self.points_cartesian['inscribed_circle'] = fit_circle(self.drop_points_cartesian, np.argmin(self.drop_points_cartesian[:, 1]))
        
        # Convert the interesting points in pixel dimensions:
        self.points_pixels = {}
        for k, point in self.points_cartesian.items():
            if k == 'inscribed_circle':
                x =  self.origin_shift[0] - point[0] * self.scale[0]
                z =  self.origin_shift[1] - point[1] * self.scale[0]
                r = point[2] * self.scale[0]
                self.points_pixels[k] = (x, z, r)
            else:
                x = self.origin_shift[0] - point[0] * self.scale[0]
                z = self.origin_shift[1] - point[1] * self.scale[0]
                self.points_pixels[k] = (x, z)
        return
    
    def calculate_drop_volume(self, num_heights=100):
        """
        Calculate the volume of the drop from contour points by interpolating x-values 
        at specified heights.

        Args:
            contour_points (np.ndarray): Array of contour points with shape (N, 2), 
                                         where N is the number of points, and 
                                         points[:, 1] corresponds to height (z).
            num_heights (int): Number of height points to sample between 0 and max height.
        
        Returns:
            float: Calculated volume of the drop.
        """
        
        if self.drop_points_cartesian is None:
            raise ValueError("drop_points_cartesian is None. Please use calibrate to scale droplet points.")
        
        # Generate a list of heights from 0 to maximum height
        heights = np.linspace(0, np.max(self.drop_points_cartesian[:, 1]), num=num_heights)

        # Separate points into left and right based on x coordinate
        x = self.drop_points_cartesian[:, 0]
        left_points = self.drop_points_cartesian[x < 0]
        right_points = self.drop_points_cartesian[x > 0]

        # Create interpolation functions for left and right sides
        left_interpolator = interp1d(left_points[:, 1], left_points[:, 0], bounds_error=False, fill_value='extrapolate')
        right_interpolator = interp1d(right_points[:, 1], right_points[:, 0], bounds_error=False, fill_value='extrapolate')

        # Calculate diameters at each height
        left_x = left_interpolator(heights)
        right_x = right_interpolator(heights)

        # Diameter is the difference between right and left x coordinates
        diameters = right_x - left_x  # right_x is positive, left_x is negative
        areas = np.pi * (diameters / 2)**2
        
        # Remove NaN values from areas and the corresponding heights
        valid_indices = ~np.isnan(areas)  # Boolean mask for non-NaN areas
        heights = heights[valid_indices]
        areas = areas[valid_indices]

        # Calculate the volume using Simpson's rule
        self.drop_volume = simpson(y=areas, x=heights)
        return self.drop_volume
    
    
    def plot_analysis(self):
        # Create a figure and plot the original ROI image
        fig, ax = plt.subplots()
        ax.imshow(self.roi_image)  # Display the original ROI image
    
    
        if self.tip_pixels is not None:
            # Overlay the tip points as a polygone
            # Define the bounding box of the tip region
            top_y = max(coord[1] for coord in self.tip_pixels)  # Highest point (tip)
            bottom_y = min(coord[1] for coord in self.tip_pixels)  # Lowest point
            left_x = min(coord[0] for coord in self.tip_pixels)  # Leftmost point
            right_x = max(coord[0] for coord in self.tip_pixels)  # Rightmost point
            
            # Define the filled area (polygon) coordinates
            tip_filled_area = [(left_x, top_y), (right_x, top_y), (right_x, bottom_y), (left_x, bottom_y)]
            
            # Create and add the filled polygon patch
            tip_patch = Polygon(tip_filled_area, closed=True, edgecolor='blue', facecolor='blue', 
                                alpha=0.4, linewidth=0, zorder=1, label='Tip')
            ax.add_patch(tip_patch)
        
        if self.drop_points is not None:
            # Plot the original contour points in Cartesian coordinates
            ax.scatter(self.drop_points[:, 0], self.drop_points[:, 1], color='red', s=1, label="Contour Points")
    
    
        if self.points_pixels is not None:
            # Plot the fitted circle
            circle_plot = plt.Circle((self.points_pixels['inscribed_circle'][0], 
                                      self.points_pixels['inscribed_circle'][1]), 
                                     self.points_pixels['inscribed_circle'][2],
                                     color='green', fill=False, linestyle='--', linewidth=1.5, label="Fitted Circle")
            plt.gca().add_patch(circle_plot)
        
            # Plot points at De
            ax.plot([self.points_pixels['max_width_left'][0], self.points_pixels['max_width_right'][0]],
                     [self.points_pixels['max_width_left'][1], self.points_pixels['max_width_right'][1]], 
                        color='orange', linestyle='--')
            x = self.points_pixels['max_width_right'][0] + (self.points_pixels['max_width_left'][0] - self.points_pixels['max_width_right'][0])/6
            y = self.points_pixels['max_width_left'][1]*0.95 
            ax.text(x, y, r'$D_{e}$',  color='orange', fontsize=14)
            
            # Plot vertical De height  
            ax.plot([self.origin_shift[0], self.origin_shift[0]], 
                     [self.origin_shift[1], self.points_pixels['Ds_left'][1]],
                        color='orange', linestyle='--')
        
            # Plot points at Ds
            ax.plot([self.points_pixels['Ds_left'][0], self.points_pixels['Ds_right'][0]],
                     [self.points_pixels['Ds_left'][1], self.points_pixels['Ds_right'][1]], 
                        color='lightblue', linestyle='--')
            x = self.points_pixels['Ds_right'][0] + (self.points_pixels['Ds_left'][0] - self.points_pixels['Ds_right'][0])/3
            y = self.points_pixels['Ds_left'][1]*0.9
            ax.text(x, y, r'$D_{s}$',  color='lightblue', fontsize=14)
        
            # Create the square
            points = [(self.points_pixels['max_width_left'][0], self.origin_shift[1]), 
                      (self.points_pixels['max_width_right'][0], self.origin_shift[1]), 
                      (self.points_pixels['max_width_right'][0], self.points_pixels['Ds_left'][1]), 
                      (self.points_pixels['max_width_left'][0], self.points_pixels['Ds_left'][1])]
            square = Polygon(points, closed=True, edgecolor='grey', linestyle='--', facecolor='none', linewidth=1.5, alpha=1, zorder=0, label='Fitting square')
            ax.add_patch(square)
        
        if self.drop_volume is not None:
            ax.text(self.origin_shift[0]*1.02, self.origin_shift[1]*0.88, 'V=' + str(int(self.drop_volume*1e9)) + ' μl', fontsize=14, color='red')
    
        # Customize plot
        ax.set_title("Fitted Circle and Bottom Points on ROI Image")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.legend(loc="upper right", )
        
        return fig, ax

    
    def extract_drop_characteristics(self, canny_thresholds={'threshold1': 50, 'threshold2': 100}, 
                                      smoothing={'window_length': 3, 'polyorder': 2}):
        """
        Streamlined function to process the image, calibrate, extract points of interest, 
        and calculate the drop volume in one step.
        
        Args:
            canny_thresholds (dict): Dictionary containing the 'threshold1' and 'threshold2' 
                                      for Canny edge detection (default values are 50 and 100).
            smoothing (dict): Dictionary containing 'window_length' and 'polyorder' for smoothing 
                              (default values are 3 and 2).
        
        Returns:
            float: The calculated drop volume.
        """
        if self.roi_image is  None:
            raise ValueError("roi_image is None, please extract roi image with extract_roi_image.")
    
        # Find drop contour using the specified Canny thresholds and smoothing parameters
        self.find_drop_contour(canny_thresholds=canny_thresholds, smoothing=smoothing)
        
        # Calibrate the Calibrate the drop contour position 
        self.calibrate()
        
        # Extract meaningful points of interest for interfacial tension calculation
        self.extract_points_of_interest()
        
        # Calculate the drop volume
        self.calculate_drop_volume()
        
        return


    def compute_surface_tension(self, constants, image_path, csv_file_path=None):
        
        delta_rho, g, needle_diameter = constants['delta_rho'], constants['g'], constants['needle_diameter']
        
        # Compute ds and de
        ds = abs(self.points_cartesian['Ds_right'][0] - self.points_cartesian['Ds_left'][0])
        de = abs(self.points_cartesian['max_width_right'][0] - self.points_cartesian['max_width_left'][0])
        
        # Compute distance error
        dist_error = 2 * (1 / self.scale[0])
    
        # Compute s and its uncertainty
        s = ds / de
        s_err = s * ((dist_error / ds)**2 + (dist_error / de)**2) ** 0.5
        s_minus, s_plus = s - s_err, s + s_err
    
        # Compute h_bar and its uncertainty
        h_bar = get_one_over_H(s)
        h_bar_minus, h_bar_plus = get_one_over_H(s_minus), get_one_over_H(s_plus)
        h_bar_err = max(abs(h_bar - h_bar_minus), abs(h_bar - h_bar_plus))
    
        # Compute surface tension and its error
        gamma = get_gamma(delta_rho, g, de, h_bar)
        gamma_err = gamma * ((h_bar_err / h_bar)**2 + 2 * (dist_error / de)**2) ** 0.5 
        

    
        # Compute Bond number and Worthington number
        r0 = self.points_cartesian['inscribed_circle'][2]
        Bo = get_bo(delta_rho, g, r0, gamma)
        Wo = get_wo(delta_rho, g, self.drop_volume, gamma, needle_diameter)
    
        print('Surface tension:', gamma, 'Bond number:', Bo, 'Worthington number:', Wo)
    
        # Extract sample name from image path
        sample_name = os.path.basename(image_path)
    
        # Prepare data for saving
        self.interfacial_values = {
            'sample_name': sample_name,
            'delta_rho': delta_rho,
            'needle_diameter': needle_diameter,
            'tip_position': self.tip_position,
            'roi_window': self.roi,
            'scale': self.scale[0],
            'ds': ds,
            'de': de,
            'dist_error': dist_error,
            's': s,
            'h_bar': h_bar,
            'gamma': gamma,
            'gamma_err': gamma_err,
            'r0': r0,
            'Bo': Bo,
            'Wo': Wo,
            'Volume': self.drop_volume,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
        # Update or append data to CSV
        if csv_file_path is not None:
            update_or_append_csv(self.interfacial_values, csv_file_path)
        return self.interfacial_values
        


def calculate_width(coordinates=None):
    
    minx = float('inf')  # Start with an infinitely large value
    if coordinates is None:
        raise ValueError("No coordinates found. Please add coordinates of interest.")

    # Group x-coordinates by z-coordinate
    width_dict = defaultdict(list)
    
    for x, z in coordinates:
        width_dict[z].append(x)
        if x < minx:
            minx = x

    # Calculate width for each z position
    widths = []
    
    for z, x_coords in width_dict.items():
        if len(x_coords) > 1:  # Only calculate width if there are at least two points
            width = max(x_coords) - min(x_coords) + 1
            widths.append(width)

    if not widths:
        raise ValueError("No valid widths found.")
    
    if np.std(widths) < 1:
        error = 1
    else:
        error = np.std(widths)
    
    return (np.mean(widths), error)


def get_scale(measured_dist=None, known_dist=None):
    """
    Calculate the average scale and standard deviation in pixels per millimeter based on the widths.

    Args:
        measured_dist (tuple): Measured distance and corresponding error.
        known_dist (float): Known distance.

    Returns:
        tuple: Average scale and standard deviation of the scale.
    """
    if measured_dist is None:
        raise ValueError("No measured distance provided.")
    
    if known_dist is None:
        raise ValueError("No known distance provided.")

    # Calculate average width and standard deviation
    ave_width, err_width = measured_dist[0], measured_dist[1]

    # Calculate scale
    scale =  ave_width / known_dist
    
    # Propagate the error using standard error propagation: 
    # Δ(scale) = (Δ(width) / width) * scale
    error = (err_width / ave_width) * scale if ave_width != 0 else 0

    return scale, error



def order_points(points):
    """
    Orders points in a clockwise manner and moves the point closest to z=0 to the beginning.
    
    Parameters:
    points (ndarray): Array of (x, z) coordinates.

    Returns:
    ndarray: Ordered points.
    """

    # Calculate centroid (handling division by zero)
    M = cv2.moments(points.astype(np.float32))  # Ensure correct dtype
    if M["m00"] == 0:
        raise ValueError("Centroid calculation failed due to zero area.")

    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]

    # Sort points based on angle relative to the centroid
    ordered_points = sorted(points, key=lambda p: np.arctan2(p[1] - cY, p[0] - cX))

    # Convert back to a NumPy array
    ordered_points = np.array(ordered_points)

    # Find the index of the point closest to z=0 in the ordered array
    new_index = np.argmin(ordered_points[:, 1])

    # Roll so that this point is first
    ordered_points = np.roll(ordered_points, -new_index, axis=0)

    return ordered_points



def smooth_savgol(points, window_length=5, polyorder=2):
    """
    Smooth the contour points using the Savitzky-Golay filter.

    Parameters:
    - points: numpy array of shape (N, 2), where each row is (x, z).
    - window_length: int, length of the filter window (must be odd).
    - polyorder: int, order of the polynomial used to fit the samples.

    Returns:
    - smoothed_points: numpy array of shape (N, 2), the smoothed contour points.
    """
    smoothed_x = savgol_filter(points[:, 0], window_length=window_length, polyorder=polyorder)
    smoothed_z = savgol_filter(points[:, 1], window_length=window_length, polyorder=polyorder)
    return np.column_stack((smoothed_x, smoothed_z))


def circle_distance(params, points):
    x0, z0, R = params
    distances = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - z0)**2)
    return distances - R


def fit_circle(points, origin_index, proportion=0.15):
    """
    Fit a circle to a subset of contour points based on the selected region near the origin index.

    Args:
        points (numpy.ndarray): 2D array of points (x, z) representing contour points.
        origin_index (int): Index of the origin point in the contour.
        proportion (float, optional): Proportion of points to consider from the bottom of the contour, default is 0.15.

    Returns:
        tuple: Fitted circle's center (x0, z0) and radius (r0).
    """
    
    # Select the bottom 'proportion' of the contour points, centered around the origin_index
    number_of_points = int(proportion * len(points))  # Number of points to select based on the proportion

    # Select points around the origin_index (centered on it)
    bottom_points = points[int(origin_index - number_of_points / 2):int(origin_index + number_of_points / 2)]
    
    # Initial guess for the circle's center (mean of the selected bottom points) and radius
    x0_init, z0_init = np.mean(bottom_points, axis=0)  # Mean of x and z coordinates for the initial center
    R0_init = np.mean(np.sqrt((bottom_points[:, 0] - x0_init) ** 2 + (bottom_points[:, 1] - z0_init) ** 2))  # Mean distance (radius) from center
    
    # Initial parameters for circle fitting: [x0_init, z0_init, R0_init]
    initial_params = [x0_init, z0_init, R0_init]

    # Fit the circle using least squares optimization based on the distance from points to the circle
    fitted_params, _ = leastsq(circle_distance, initial_params, args=(bottom_points,))
    
    # Unpack the fitted parameters (center and radius of the circle)
    x0, z0, r0 = fitted_params
    
    # Display the fitted circle parameters
    print(f"Fitted Circle Center: (x0: {x0:.2f}, z0: {z0:.2f}), Radius: r0: {r0:.2f}")
    
    # Return the fitted circle's parameters
    return (x0, z0, r0)


def smallest_bounding_square(points):
    """
    Computes the smallest bounding square that encloses the key points 
    of a given contour in Cartesian coordinates.

    Parameters:
        points (ndarray): Array of (x, z) coordinates representing the contour.

    Returns:
        dict: Coordinates of the bounding square's corners.
    """

    # Identify the three leftmost and rightmost points based on x-coordinates
    min_x_points = points[np.argsort(points[:, 0])[:3]]
    max_x_points = points[np.argsort(points[:, 0])[-3:]]

    # Compute the average x and z values for the leftmost and rightmost points
    avg_min_x, z_at_avg_min = np.mean(min_x_points, axis=0)
    avg_max_x, z_at_avg_max = np.mean(max_x_points, axis=0)

    # Determine the side length of the square to enclose the range of x and z values
    side_length = max(avg_max_x - avg_min_x, z_at_avg_max - z_at_avg_min)

    # Define the coordinates of the bounding square
    return {"bottom_left": (avg_min_x, 0),
            "bottom_right": (avg_min_x + side_length, 0),
            "top_left": (avg_min_x, side_length),
            "top_right": (avg_min_x + side_length, side_length),
            "max_width_left": (avg_min_x, (z_at_avg_min+z_at_avg_max)/2),
            "max_width_right": (avg_max_x, (z_at_avg_min+z_at_avg_max)/2),
            }


def extrapolate_x_at_z(point1, point2, target_z):
    """
    Extrapolate the x value at a given z using two points.

    Parameters:
    - point1: tuple, the first point (x1, z1)
    - point2: tuple, the second point (x2, z2)
    - target_z: float, the target z value for which to extrapolate x

    Returns:
    - extrapolated_x: float, the extrapolated x value at target z
    """
    x1, z1 = point1
    x2, z2 = point2
    
    # Check if z1 and z2 are the same
    if z1 == z2:
        raise ValueError("z1 and z2 cannot be the same value for extrapolation.")

    # Calculate the slope of the line connecting the two points
    slope = (x2 - x1) / (z2 - z1)
    
    # Use the slope to extrapolate x at the target_z
    extrapolated_x = x1 + slope * (target_z - z1)

    return extrapolated_x



def closest_points_to_z(points, z_target):
    """
    Extract the coordinates of the two points for which the z values 
    are closest to a given z_target.

    Parameters:
    - points: numpy array of shape (N, 2) where each row is a point (x, z).
    - z_target: float, the target z value.

    Returns:
    - closest_points: numpy array of shape (2, 2), coordinates of the closest points.
    """
    # Extract z values from points
    z_contour = points[:, 1]
    
    # Calculate the absolute differences from the target z value
    differences = z_contour - z_target

    # Find the indices of the closest point below and above the target
    below_indices = np.where(differences < 0)[0]  # Indices of points below z_target
    above_indices = np.where(differences > 0)[0]  # Indices of points above z_target

    if below_indices.size == 0 or above_indices.size == 0:
        raise ValueError("There must be at least one point above and one point below the target z value.")

    # Find the closest point below z_target
    closest_below_index = below_indices[np.argmax(z_contour[below_indices])]  # Maximum z value below z_target

    # Find the closest point above z_target
    closest_above_index = above_indices[np.argmin(z_contour[above_indices])]  # Minimum z value above z_target

    # Extract the corresponding coordinates
    closest_points = points[[closest_below_index, closest_above_index]]
    
    return closest_points


def get_intersections_at_z(points, z):
    """
    Finds the x-coordinates where the contour intersects a given z level, 
    separately for the left (x < 0) and right (x >= 0) sides.

    Parameters:
        points (ndarray): Array of (x, z) coordinates representing the contour.
        z (float): The z-coordinate at which to find intersections.

    Returns:
        dict: Intersection points {'left': (x_left, z), 'right': (x_right, z)}.
    """

    # Split the contour into left (x < 0) and right (x >= 0) segments
    left_points = points[points[:, 0] < 0]
    right_points = points[points[:, 0] >= 0]

    # Find the two closest points to the given z level and extrapolate x for each side
    left_x = extrapolate_x_at_z(*closest_points_to_z(left_points, z), z)
    right_x = extrapolate_x_at_z(*closest_points_to_z(right_points, z), z)

    return {"left":(left_x, z),"right": (right_x, z)}



def get_one_over_H(s):
    '''
    Function to calculate 1/H based on the equation provided in:
    Drelich J, Fang C, White CL (2002) Measurement of interfacial tension in fluid-fluid systems. Encycl Surf Colloid Sci 3:3158–3163
    The function takes a value 's' as input and computes the corresponding 1/H.
    The formula used depends on the value of 's' and follows different conditions.
    '''
    # Check if the input value 's' is within the valid range for the calculations.
    if s >= 0.3:
        if s <= 0.4:
            # Apply the formula for the range 0.3 <= s <= 0.4
            one_over_H = (0.34074/s**(2.52303)) + 123.9495*s**5 - 72.82991*s**4 + 0.01320*s**3 - 3.38210*s**2 + 5.52969*s - 1.07260
        elif s <= 0.46:
            # Apply the formula for the range 0.4 < s <= 0.46
            one_over_H = (0.32720/s**(2.56651)) - 0.97553*s**2 + 0.84059*s - 0.18069
        elif s <= 0.590:
            # Apply the formula for the range 0.46 < s <= 0.590
            one_over_H = (0.31968/s**(2.59725)) - 0.46898*s**2 + 0.50059*s - 0.13261
        elif s <= 0.680:
            # Apply the formula for the range 0.590 < s <= 0.680
            one_over_H = (0.31522/s**(2.62435)) - 0.11714*s**2 + 0.15756*s - 0.05285
        elif s <= 0.9:
            # Apply the formula for the range 0.680 < s <= 0.9
            one_over_H = (0.31345/s**(2.64267)) - 0.09155*s**2 + 0.14701*s - 0.05877
        elif s <= 1.0:
            # Apply the formula for the range 0.9 < s <= 1.0
            one_over_H = (0.30715/s**(2.84636)) - 0.69116*s**3 + 1.08315*s**2 - 0.18341*s - 0.20970
        else:
            raise ValueError("Method can't be used for S values > 1. s =", s)
            
    else:
        raise ValueError("Method can't be used for S values < 0.3. s =", s)
    
    return one_over_H


def get_gamma(delta_rho, g, de, h_bar):
    return (delta_rho * g * de**2) * h_bar


def get_bo(delta_rho, g, r0, gamma):
    return delta_rho * g * r0**2 / gamma


def get_wo(delta_rho, g, V, gamma, tip_diameter):
    #Berry, J. D., Neeson, M. J., Dagastine, R. R., Chan, D. Y. C. & Tabor, R. F. Measurement of surface and interfacial tension using pendant drop tensiometry. Journal of Colloid and Interface Science 454, 226–237 (2015).
    return delta_rho * g * V / (np.pi * gamma * tip_diameter)



def update_or_append_csv(data, csv_file_path):
    # Function to check if a sample exists and update its row, or append a new row
    fieldnames = data.keys()
    # Check if file exists
    file_exists = os.path.isfile(csv_file_path)

    # Read existing data
    rows = []
    sample_found = False
    if file_exists:
        with open(csv_file_path, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['sample_name'] == data['sample_name']:
                    row.update(data)  # Update row with new data
                    sample_found = True
                rows.append(row)
    
    # If sample not found, add it
    if not sample_found:
        rows.append(data)

    # Write updated data back to file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
