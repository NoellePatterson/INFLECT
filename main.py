"""
INFLECT (INFLection-based Elevations from Channel Topography) Method

Primary implementtion script.

Noelle Patterson, USU
Funded by CIROH (Cooperative Institute for Research to Operations) 
2024/2025
"""

import os
import geopandas as gpd
import pandas as pd
import rasterio
import ast
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
import numpy as np
from matplotlib import pyplot as plt
# import dataretrieval.nwis as nwis
from datetime import datetime
import sys
from analysis import calc_dwdh, calc_derivatives_aggregate
from visualization import output_record, plot_bankfull_increments, plot_longitudinal_profile, transect_plot, plot_inflections
from spatial_analysis import create_bankfull_pts

reach_name = 'Mad_lower' # Choose a reach name for output file identification

# Steps for bankfull analysis:
# 1. Identify benchmark bankfull using inundation rasters (Analysis.py -> id_benchmark_bankfull)
# 2. Measure channel width along a depth interval for each cross-section (Analysis.py -> calc_dwdh)
# 3. Calculate first and second order derivatives of the channel widths to identify topographic bankfull (Analysis.py -> calc_derivatives)
# 4. Post-processing: plot results (Visualization.py -> plot_bankfull_increments, plot_longitudinal_bf)

# Specify input data file paths in correct input folder directories
dem_fp = 'data_inputs/dem/dem.tif' # file in 'data_inputs/dem/...' folder
thalweg_fp = 'data_inputs/thalweg/Thalweg.shp' # file in 'data_inputs/thalweg/...' folder
cross_sections_fp = 'data_inputs/cross_sections/transects_lower_long.shp' # file in 'data_inputs/cross_sections/...' folder

# Create output folders if needed
if not os.path.exists('data_outputs/{}'.format(reach_name)):
    os.makedirs('data_outputs/{}'.format(reach_name))
if not os.path.exists('data_outputs/{}/transect_plots'.format(reach_name)):
    os.makedirs('data_outputs/{}/transect_plots'.format(reach_name))
if not os.path.exists('data_outputs/{}/second_order_roc'.format(reach_name)):
    os.makedirs('data_outputs/{}/second_order_roc'.format(reach_name))
if not os.path.exists('data_outputs/{}/all_widths'.format(reach_name)):
    os.makedirs('data_outputs/{}/all_widths'.format(reach_name))
if not os.path.exists('data_outputs/{}/spatial'.format(reach_name)):
    os.makedirs('data_outputs/{}/spatial'.format(reach_name))

# Read in data: dem, thalweg, and cross-sections 
dem = rasterio.open(dem_fp)
thalweg = gpd.read_file(thalweg_fp)
cross_sections = gpd.read_file(cross_sections_fp)

# Set INFLECT algorithm parameters
plot_interval = 1 # set plotting interval along transect in units of meters
d_interval = 10/100 # Set intervals to step up in depth (in units meters). Default is 10cm intervals
slope_window = 10 # Set window size for calculating slope for derivatives
lower_bound = 5 # Set lower vertical boundary for inflection id within cross-section, in units of d_interval. Default 5 = 50cm
upper_bound = 100 # Set upper vertical boundary for inflection id within cross-section, in units of d_interval. Default 100 = 10m
spatial_plot_interval = 0.5 # interval for finding inflection elevation along cross_sections.
width_calc_method = 'partial' # 'continuous' 'partial' - choose from either partial additive widths or continuous-only methods of width calculation

# Uncomment Analysis functions to run

# all_widths_df = calc_dwdh(reach_name, cross_sections, dem, plot_interval, d_interval, width_calc_method) # calc widths array for each cross-section
# calc_derivatives_aggregate(reach_name, d_interval, all_widths_df, slope_window)
# output_record(reach_name, slope_window, d_interval, lower_bound, upper_bound, width_calc_method)

# Plotting functions
plot_longitudinal_profile(reach_name, dem, cross_sections, plot_interval)
plot_bankfull_increments(reach_name, d_interval)
transect_plot(cross_sections, dem, plot_interval, d_interval, reach_name)
plot_inflections(d_interval, reach_name)
# Spatial analysis
create_bankfull_pts(cross_sections, dem, thalweg, d_interval, spatial_plot_interval, reach_name)