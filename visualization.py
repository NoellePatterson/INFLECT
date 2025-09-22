import pandas as pd
import numpy as np
from numpy import nan
import re
import glob
import geopandas as gpd
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_longitudinal_profile(reach_name, dem, cross_sections, plot_interval):
    # Extract and detrend thalweg for plotting
    thalweg_distances = []
    thalweg_line = []
    for cross_sections_index, cross_sections_row in cross_sections.iterrows():
        line = gpd.GeoDataFrame({'geometry': [cross_sections_row['geometry']]}, crs=cross_sections.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = cross_sections_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=cross_sections.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        thalweg = min(elevs) # track this for use later in detrending
        thalweg_line.append(thalweg)
        # find station coordinates at thalweg
        if cross_sections_index > 0: # measure distances for all but first (most upstream) transect
            thalweg_index = elevs.index(thalweg)
            thalweg_coords = stations.geometry[thalweg_index]
            # Get distance from thalweg to next thalweg
            next_transect = cross_sections.iloc[cross_sections_index - 1]
            next_line = gpd.GeoDataFrame({'geometry': [next_transect['geometry']]}, crs=cross_sections.crs)
            next_tot_len = next_line.length
            next_distances = np.arange(0, next_tot_len[0], plot_interval) 
            next_stations = next_transect['geometry'].interpolate(next_distances) # specify stations in transect based on plotting interval
            next_stations = gpd.GeoDataFrame(geometry=next_stations, crs=cross_sections.crs)
            next_elevs = list(dem.sample([(point.x, point.y) for point in next_stations.geometry]))
            next_thalweg = min(next_elevs)
            next_thalweg_index = next_elevs.index(next_thalweg)
            next_thalweg_coords = next_stations.geometry[next_thalweg_index]
            thalweg_distance = next_thalweg_coords.distance(thalweg_coords) # distance to next thalweg, in meters
        else: 
            thalweg_distance = 0
        if cross_sections_index == 0:
            thalweg_distances.append(thalweg_distance)
        else:
            thalweg_distances.append(thalweg_distance + thalweg_distances[cross_sections_index-1])

    thalweg_detrend = []
    x_vals_thalweg = np.arange(0, len(thalweg_line))
    x = np.array(x_vals_thalweg).reshape(-1, 1)
    y = np.array(thalweg_line)
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope =  slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from thalwegs
    for index, val in enumerate(thalweg_line):
        thalweg_detrend.append(val - fit_slope[index])
    # Plot logitudinal profile
    # breakpoint()
    fig, ax = plt.subplots()
    plt.xlabel('Cross-sections from upstream to downstream (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Logitudinal profile, {}'.format(reach_name))
    plt.plot(thalweg_distances, thalweg_line, color='grey', label='Thalweg (detrended)')
    plt.plot(thalweg_distances, fit_slope + intercept)
    plt.legend(loc='upper right')
    plt.savefig('data_outputs/{}/Bankfull_longitudinals'.format(reach_name))
    plt.close()

def plot_bankfull_increments(reach_name, d_interval, plot_ylim):
    agg_bankfull = pd.read_csv('data/data_outputs/{}/max_inflections.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data/data_outputs/{}/all_widths.csv'.format(reach_name))
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'widths'] = eval(row['widths'])

    # Detrend widths before plotting based on thalweg elevation, and start plotting point based on detrend
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope] # unnest the array
    
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style
    fig, ax = plt.subplots()
    plt.ylabel('Channel width (m)')
    plt.xlabel('Detrended elevation (m)')
    plt.title('Incremental channel widths for {}'.format(reach_name))

    for index, row in all_widths_df.iterrows(): 
        row = row['widths']
        x_len = round(len(row) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        # apply detrend shift to xvals
        x_vals = [x_val - fit_slope[index] - intercept for x_val in x_vals]
        plt.plot(x_vals, row, alpha=0.3, color=cmap(norm(index)), linewidth=0.75) 
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)")
    plt.savefig('data/data_outputs/{}/all_widths.jpeg'.format(reach_name), dpi=400)
    plt.close()

    # Plot average and bounds on all widths
    # calc element-wise avg, 25th, & 75th percentile of each width increment
    bankfull_benchmark = bankfull_benchmark['benchmark_bankfull_ams_detrend']
    bankfull_topo = bankfull_topo['bankfull']
    benchmark_25 = np.nanpercentile(bankfull_benchmark, 25)
    benchmark_75 = np.nanpercentile(bankfull_benchmark, 75)
    topo_25 = np.nanpercentile(bankfull_topo, 25)
    topo_75 = np.nanpercentile(bankfull_topo, 75)

    fig, ax = plt.subplots()
    plt.xlabel('Height above sea level (m)')
    plt.ylabel('Channel width (m)')
    plt.title('Incremental channel top widths for {}'.format(reach_name))
    # Prepare widths for plotting
    max_len = max(all_widths_df['widths'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded'] = all_widths_df['widths'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df = pd.DataFrame(all_widths_df['widths_padded'].tolist())
    transect_50 = padded_df.apply(lambda row: np.nanmedian(row), axis=0)
    transect_25 = padded_df.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75 = padded_df.apply(lambda row: np.nanpercentile(row, 75), axis=0)

    x_len = round(len(transect_50) * d_interval, 4)
    x_vals = np.arange(0, x_len, d_interval)
    if reach_name == 'Scotia':
        plt.xlim(plot_ylim) # truncate unneeded values from plot
    if reach_name == 'Miranda':
        plt.xlim(60, 80)
    if reach_name == 'Leggett':
        plt.xlim(plot_ylim)
    plt.plot(x_vals, transect_50, color='black', label='Width/height median')
    plt.plot(x_vals, transect_25, color='blue', label='Width/height 25-75%')
    plt.plot(x_vals, transect_75, color='blue')
    plt.legend(loc='center right')

    plt.legend()

    # Detrended, aggregated cross-sections using padded-zeros approach
    all_widths_df['widths_detrend'] = [[] for _ in range(len(all_widths_df))] 
    # Loop through all_widths
    for index, row in all_widths_df.iterrows():
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        if offset_int < 0: # most likely case, downstream xsections are lower elevation than furthest upstream
            # populate new column of df with width values
            all_widths_df.loc[index, 'widths_detrend'].extend([0] * abs(offset_int) + row['widths']) # add zeros to beginning of widths list. Need to unnest when using.
        elif offset_int > 0: # this probably won't come up
            all_widths_df.loc[index, 'widths_detrend'].extend(row[abs(offset_int):])
        else:
            all_widths_df.loc[index, 'widths_detrend'].extend(row['widths'])

    # Once all offsets applied, use zero-padding aggregation method just like with non-detrended widths.
    max_len = max(all_widths_df['widths_detrend'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded_detrend'] = all_widths_df['widths_detrend'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df_detrend = pd.DataFrame(all_widths_df['widths_padded_detrend'].tolist())
    transect_50_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 75), axis=0)
    plot_df = pd.DataFrame({'50th':transect_50_detrend, '25th':transect_25_detrend, '75th':transect_75_detrend})
    plot_df.to_csv('data_outputs/{}/width_elev_plotlines.csv'.format(reach_name))

def stacked_width_plots(d_interval):
    # Bring in all plot lines from upper, mid, lower
    upper_plotlines = pd.read_csv('data/data_outputs/Leggett/width_elev_plotlines.csv')
    mid_plotlines = pd.read_csv('data/data_outputs/Miranda/width_elev_plotlines.csv')
    lower_plotlines = pd.read_csv('data/data_outputs/Scotia/width_elev_plotlines.csv')
    # Determine where to begin plotting based on where median line goes above zero
    def start_plot(line_25th):
        for index, value in line_25th.items():
            if value > 0:
                return index
    upper_plot_start = start_plot(upper_plotlines['25th'])
    upper_plotlines = upper_plotlines[upper_plot_start:].reset_index()
    middle_plot_start = start_plot(mid_plotlines['25th'])
    mid_plotlines = mid_plotlines[middle_plot_start:].reset_index()
    lower_plot_start = start_plot(lower_plotlines['25th'])
    lower_plotlines = lower_plotlines[lower_plot_start:].reset_index()

    def get_x_vals(y_vals):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
    x_upper = get_x_vals(upper_plotlines['50th'][:110])
    x_mid = get_x_vals(mid_plotlines['50th'][:110])
    x_lower = get_x_vals(lower_plotlines['50th'][:110])
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(x_upper, upper_plotlines['50th'][:110], color='orange', label='upper reach')
    plt.plot(x_mid, mid_plotlines['50th'][:110], color='green',label='middle reach')
    plt.plot(x_lower, lower_plotlines['50th'][:110], color='blue',label='lower reach')
    plt.xlabel('Relative elevation (m)')
    plt.ylabel('Channel width (m)')
    plt.legend()

    plt.savefig('data/data_outputs/stacked_width_elev.jpeg')
    return

def transect_plot(cross_sections, dem, plot_interval, d_interval, reach_name):
    # topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    inflections = pd.read_csv('data_outputs/{}/max_inflections_aggregate.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))

    # Use thalweg to detrend elevation on y-axes for transect plotting. Don't remove intercept (keep at elevation) 
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    
    # for cross_section in cross_sections:
    for transects_index, transects_row in cross_sections.iterrows():
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=cross_sections.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=cross_sections.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # Arrange points together for plotting
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        fig = plt.figure(figsize=(6,8))
        plt.plot(distances, elevs, color='black', linestyle='-', label='Cross section')
        for index, x in enumerate(inflections['pos_inflections']):
            if index == 0:
                plt.axhline(x*d_interval, color='red', label='positive inflections', linewidth=2)
            else:
                plt.axhline(x*d_interval, color='red', linewidth=2)
        for index, x in enumerate(inflections['neg_inflections']):
            if index == 0:
                plt.axhline(x*d_interval, color='blue', label='negative inflections', linewidth=2)
            else:
                plt.axhline(x*d_interval, color='blue', linewidth=2)

        plt.xlabel('Cross section distance (meters)', fontsize=16)
        plt.ylabel('Elevation (meters)', fontsize=16)
        plt.legend(fontsize=16)
        # increase font size for axes and labels
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.ylim(top=45)
        # Make the bottom of the ylim fall a meter below the lowest point in cross section
        plt.tight_layout()
        plt.savefig('data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, transects_index))
        plt.close()

def plot_wd_and_xsections(reach_name, d_interval, plot_ylim, transects, dem, plot_interval):
    bankfull_topo = pd.read_csv('data/data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    bankfull_benchmark = pd.read_csv('data/data_outputs/{}/bankfull_benchmark_detrend.csv'.format(reach_name))
    aggregate_topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_aggregate_elevation.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data/data_outputs/{}/all_widths.csv'.format(reach_name))
    median_bf_topo = np.nanmedian(bankfull_topo['bankfull'])
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'widths'] = eval(row['widths'])
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style


    # Add a second panel with cross-section
    # Loop through cross-sections and plot each one
    for transects_index, transects_row in transects.iterrows():
        current_bf_topo = bankfull_topo['bankfull'][transects_index]
        # 1. Plot spaghetti lines on first plot panel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.set_ylabel('Height above sea level (m)')
        ax1.set_xlabel('Channel width (m)')
        ax1.set_title('Channel width/height ratios for {}'.format(reach_name))
        if reach_name == 'Leggett':
            ax1.set_ylim((220,270))
        elif reach_name == 'Miranda':
            ax1.set_ylim((62.5, 80))
        elif reach_name == 'Scotia':
            ax1.set_ylim(10,40)

        for index, row in all_widths_df.iterrows(): 
            row = row['widths']
            x_len = round(len(row) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            if index == transects_index:
                ax1.plot(row, x_vals, alpha=1, color='red', linewidth=1)
            else:       
                ax1.plot(row, x_vals, alpha=0.3, color=cmap(norm(index)), linewidth=0.75) # Plot w elevation on y axis
        ax1.axhline(median_bf_topo, label='Median topographic bankfull'.format(str(median_bf_topo)), color='black', linewidth=0.75)
        ax1.axhline(current_bf_topo, label='Cross-section topographic bankfull'.format(str(current_bf_topo)), color='red', linewidth=0.75)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Set array to avoid warnings
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label("Downstream distance (m)")

        # 2. Plot cross-section on second panel
        
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs)
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval)
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        normalized_elevs = elevs
        def get_x_vals(y_vals):
                x_len = round(len(y_vals) * d_interval, 4)
                x_vals = np.arange(0, x_len, d_interval)
                return(x_vals)
        min_y = min(normalized_elevs)
        ax2.plot(distances, normalized_elevs, color='black', linestyle='-', label='Cross section')
        ax2.axhline(median_bf_topo, label='Median topographic bankfull'.format(str(median_bf_topo)), color='black', linewidth=0.75)
        ax2.axhline(current_bf_topo, label='Cross-section topographic bankfull'.format(str(current_bf_topo)), color='red', linewidth=0.75)
        ax2.legend()
        ax2.set_ylim(62.5, 80)
        ax2.set_title('Cross section {}'.format(str(transects_index)))
        plt.savefig('data/data_outputs/{}/dw_xs_plots/{}.jpeg'.format(reach_name, transects_index), dpi=400)
        plt.close()

def multi_panel_plot(reach_name, transects, dem, plot_interval, d_interval, bankfull_boundary):
    topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        intersect_pts = line.geometry.intersection(bankfull_boundary)
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # Base all depths on 0-elevation
        normalized_elevs = elevs

        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        # Be aware, Scotia transects size exceeds limits for figures saved in one folder (no warning issued)
        bankfull_z = list(dem.sample(coords))
        bankfull_z_plot = [bankfull_z[0][0], bankfull_z[1][0]] # elevation of benchmark bankfull for plotting
        bankfull_z_plot = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies

        # bring in topo bankfull for plotting
        current_topo_bankfull = topo_bankfull['bankfull'][transects_index]

        # Arrange points together for plotting
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        min_y = min(normalized_elevs)
        fig = plt.figure(figsize=(8,8))
        plt.plot(distances, normalized_elevs, color='black', linestyle='-', label='Cross section')
        plt.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='Topographic-derived bankfull')
        plt.axhline(bankfull_z_plot, color='red', linestyle='-', label='Benchmark bankfull')
        plt.xlabel('Cross section distance (meters)', fontsize=12)
        plt.ylabel('Elevation (meters)', fontsize=12)
        plt.legend(fontsize=12)
        # increase font size for axes and labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Make the bottom of the ylim fall a meter below the lowest point in cross section
        if reach_name == 'Scotia':
            plt.ylim((min_y-1),30)
        elif reach_name == 'Miranda':
            plt.ylim((min_y-1),80)
        elif reach_name == 'Leggett':
            plt.ylim((min_y-1),250)
        plt.tight_layout()
        plt.savefig('data/data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, transects_index))
        plt.close()
        # pdb.set_trace()

def plot_inflections(d_interval, reach_name):
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    inflections = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    def get_x_vals(y_vals, d_interval):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
    # bring in 2nd derivative files
    inflections_fp = glob.glob('data_outputs/{}/second_order_roc/*'.format(reach_name))
    # Sort the inflections files numerically
    def extract_num(path):
        match = re.search(r'\d+', path)
        return int(match.group()) if match else np.nan 
    inflections_fp_sorted = sorted(inflections_fp, key=extract_num)
    # bring in aggregated inflections array for plotting
    inflections_array_agg = pd.read_csv('data_outputs/{}/inflections_array_agg.csv'.format(reach_name))
    # Use thalweg elevs to detrend 2nd derivatives. Don't remove intercept (keep at elevation) 
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]

    # Set up plot and create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(inflections_fp_sorted)-1)
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.ylabel(r'Inflection magnitude $(1/m)$', fontsize=24)
    plt.xlabel('Detrended elevation (m)', fontsize=24)
    plt.title('Cross section width inflections for {}'.format(reach_name), fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # Set x-axis labels to show only integer values
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    # loop through files and plot
    for index, inflection_fp in enumerate(inflections_fp_sorted): 
        inflection = pd.read_csv(inflection_fp)
        # detrend inflections so they all plot at the same starting point
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        # if index >20:
        #     continue
            # first inflection plots all wonky, skip it
        if offset_int < 0:
            inflection = [0] * abs(offset_int) + inflection['ddw'].tolist()
        else: # Only other case is no detrend (first transect)
            inflection = inflection
        # plot all inflections spaghetti style
        x_vals = get_x_vals(inflection, d_interval)
        # if reach_name == 'Leggett':
        #     x_vals = x_vals - 220 # Hardcode in a zero-start for the elevation axis
        plt.plot(x_vals, inflection, alpha=0.5, color=cmap(norm(index)), linewidth=1.25) 
    for index, x in enumerate(inflections['pos_inflections']):
        if index == 0:
            plt.axvline(x*d_interval, color='red', label='positive inflections', linewidth=2)
        else:
            plt.axvline(x*d_interval, color='red', linewidth=2)
    for index, x in enumerate(inflections['neg_inflections']):
        if index == 0:
            plt.axvline(x*d_interval, color='blue', label='negative inflections', linewidth=2)
        else:
            plt.axvline(x*d_interval, color='blue', linewidth=2)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)", fontsize=20)
    # overlay aggregate inflections
    x_vals_overlay = get_x_vals(inflections_array_agg, d_interval)
    # if reach_name == 'Leggett':
    #         x_vals_overlay = x_vals_overlay - 220 # Hardcode in a zero-start for the elevation axis
    plt.plot(x_vals_overlay, inflections_array_agg, color='black', linewidth=1.5)
    plt.savefig('data_outputs/{}/inflections_all.jpeg'.format(reach_name))
    return

def output_record(reach_name, slope_window, d_interval, lower_bound, upper_bound, width_calc_method):
    inflections = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    # Consolidate 'pos_inflections' and 'neg_inflections' columns into single lists
    pos_inflections_all = []
    neg_inflections_all = []
    for val in inflections['pos_inflections']:
        pos_inflections_all.append(val)
    for val in inflections['neg_inflections']:
        neg_inflections_all.append(val)
    record_df = pd.DataFrame({'positive inflections':[pos_inflections_all], 'negative inflections':[neg_inflections_all], \
                              'width calc method':[width_calc_method], 'derivative slope_window': [slope_window], 'width calc interval (m)': [d_interval], 'lower_search_bound': [lower_bound], \
                                'upper_search_bound': [upper_bound]})
    record_df.to_csv('data_outputs/{}/Summary_results.csv'.format(reach_name))
