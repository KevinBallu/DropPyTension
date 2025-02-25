#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import os
os.chdir('/Users/kevin/Desktop/DropPyTension')


# Importing the necessary parts of the package
from DropPyTension import ImageProcessor, setup_environment, setup_constants


#%% Load the image and declare problem variables

# Highest folder containing everything we will work on
project_dir = '/Users/kevin/Desktop/DropPyTension/example'
# Path from project to image
image_filename = './example_images/water_in_hexadecane_9c.jpg'
# Path were the result csv will be printed
csv_filename = './Example_pendant_drops.csv'

# Set up environment and constants
image_path, csv_file_path = setup_environment(project_dir, image_filename, csv_filename)

# User provides required constants
constants = setup_constants(
    density_difference=997-773,       # Example for hexadecane
    gravitational_acceleration=9.80665, # Gravitational acceleration in m/s²
    needle_diameter=0.718e-3          # Needle diameter in meters
)

print(constants)

# Load image
processor = ImageProcessor(image_path)
processor.draw_image(processor.image)

#%% select region of interest

# Set the Region of Interest (ROI) and update until satisfied
processor.set_roi(147, 10, 530, 470)

# Draw the ROI on the image and display it
processor.draw_roi()

# when happy extract roi image
processor.extract_roi_image()


#%% Treat the image
''' 
Either by manually performing all operations:
    
processor.make_gray_image(img=processor.roi_image)
processor.draw_image(img=processor.gray_image)

processor.make_blurry_image(img=processor.gray_image)
processor.draw_image(img=processor.blurry_image)

processor.apply_binary_threshold(img=processor.blurry_image)
'''

# Or by using the wrapper function
processor.process_roi_image()

#%% Extract tip and scale

# Set the tip position and adapt until satisfied
tip_position = 70
# Extract black pixel coordinates above the tip position
processor.extract_tip_pixels(tip_position)
# Draw the tip
processor.draw_tip()

# Extract the scale of the image
processor.get_image_scale(constants['needle_diameter'])


#%% Detect Edges and Find contours 
'''
processor.find_drop_contour(canny_thresholds = {'threshold1': 50, 'threshold2': 100}, 
                            smoothing = {'window_length':3, 'polyorder':2} )

processor.calibrate()

# Extract meaningful points
processor.extract_points_of_interest()

# Calculate drop volume
volume = processor.calculate_drop_volume()

'''

# Equivalently, use the streamlined version:
processor.extract_drop_characteristics(canny_thresholds = {'threshold1': 50, 'threshold2': 100}, 
                            smoothing = {'window_length':3, 'polyorder':2} )

#%% Plot everything

#import matplotlib.pyplot as plt

fig, ax = processor.plot_analysis()
#plt.savefig('./final_analysis.png', bbox_inches='tight', dpi=300)

#%%
# calculate the interfacial tension 
#(it will be appended in the csv automatically if csv_file_path is provided)
processor.compute_surface_tension(constants, image_path, csv_file_path)

# You can have a look at the different attributes
processor.print_attributes()