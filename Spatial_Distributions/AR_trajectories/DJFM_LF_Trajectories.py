from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset
import matplotlib
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.colors as mcols
import glob 
import colorcet as cc
import netCDF4
import cmaps
from scipy.interpolate import interp2d
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.gridspec as gridspec
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Proj
# from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
from colorspacious import cspace_converter
import pathlib
from pathlib import Path
import numpy.ma as ma
from numpy import genfromtxt
import pandas as pd
import calendar
from IPython.core.pylabtools import figsize
from scipy import stats
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

## This script by Chad Small plots the centroid trajectories for DJFM for  Non-MJO, MJO-Active, and MJO-Connected ARs. It also plots the combination as well

## non-mjo ARs

non_mjo_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_Non_MJO_ARs.csv')
non_mjo_df = non_mjo_df.drop(columns=['Unnamed: 0'])

dec=non_mjo_df[non_mjo_df['Month'] == 12]
jan=non_mjo_df[non_mjo_df['Month'] == 1]
feb=non_mjo_df[non_mjo_df['Month'] == 2]
mar=non_mjo_df[non_mjo_df['Month'] == 3]

CA_non_DJF=pd.concat([dec, jan, feb, mar], ignore_index=True)

Pac_ars_df = CA_non_DJF

num_ars = str(len(Pac_ars_df))

#plot and let's see
cm = 180
fig = plt.figure(figsize=[14, 7])
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
# Set central_longitude to 0
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title('DJFM Trajectory of Non-MJO AR Systems', fontsize=14)
ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
ax.coastlines('50m', linewidth=2, zorder=5)

# Loop through ARs
for i in range(0,len(Pac_ars_df)):
    ar_test = xr.open_dataset(Pac_ars_df['AR ID (string)'].iloc[i])

    ar_test['centroid_lon'] = ar_test['centroid_lon'].where(ar_test['centroid_lon'] != -999)
    ar_test['centroid_lat'] = ar_test['centroid_lat'].where(ar_test['centroid_lat'] != -999)

    # Manually shift longitude values to center the region of interest
    lon_values = ar_test['centroid_lon'].values - 180
    lat_values = ar_test['centroid_lat'].values

    if np.max(lon_values) > 180:
        # Split trajectory into two parts
        split_index = np.argmax(lon_values > 180)
        lon_values_part1 = lon_values[:split_index]
        lat_values_part1 = lat_values[:split_index]

        # Switch coordinates to -180 to 180 for part2
        lon_values_part2 = lon_values[split_index:] - 360
        lat_values_part2 = lat_values[split_index:]

        # Plot part1
        ax.plot(lon_values_part1, lat_values_part1, alpha=0.3, color='gray', zorder=2)

        # Plot part2
        ax.plot(lon_values_part2, lat_values_part2, alpha=0.3, color='gray', zorder=2)

    else:
        # Plot the trajectory directly
        ax.plot(lon_values, lat_values, alpha=0.3, color='gray', zorder=2)

    # Plot the start and end points
    ax.scatter(lon_values[-1], lat_values[-1], alpha=0.2, color='red', zorder=3, label='End')
    ax.scatter(lon_values[0], lat_values[0], alpha=0.2, color='blue', zorder=3, label='Start')
    # plt.legend()  

# plt.legend()    

# Set the desired extent manually
ax.set_xlim([-110, 80])
ax.set_ylim([-15, 75])


# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5)
gl.ypadding = 5
gl.xpadding = 5
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 45)[::1])
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 45)[::1])

fig.text(0.50, 0.02, 'This composite constitutes '+num_ars+' AR systems at hourly resolution from June 2000 through June 2024',
         horizontalalignment='center', wrap=True)
fig.savefig("/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/Figures/Trajectories/DJFM_Non_MJO_Trajectory_Spag.png", dpi=350, bbox_inches='tight') 
# fig.show()

## MJO-Active ARs

mjo_active_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_MJO_Active_ARs.csv')
mjo_active_df = mjo_active_df.drop(columns=['Unnamed: 0'])

dec=mjo_active_df[mjo_active_df['Month'] == 12]
jan=mjo_active_df[mjo_active_df['Month'] == 1]
feb=mjo_active_df[mjo_active_df['Month'] == 2]
mar=mjo_active_df[mjo_active_df['Month'] == 3]

CA_non_DJF=pd.concat([dec, jan, feb, mar], ignore_index=True)

Pac_ars_df = CA_non_DJF

num_ars = str(len(Pac_ars_df))
#plot and let's see
cm = 180
fig = plt.figure(figsize=[14, 7])
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
# Set central_longitude to 0
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title('DJFM Trajectory of MJO-Active AR Systems', fontsize=14)
ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
ax.coastlines('50m', linewidth=2, zorder=5)

# Loop through ARs
for i in range(0,len(Pac_ars_df)):
    ar_test = xr.open_dataset(Pac_ars_df['AR ID (string)'].iloc[i])

    ar_test['centroid_lon'] = ar_test['centroid_lon'].where(ar_test['centroid_lon'] != -999)
    ar_test['centroid_lat'] = ar_test['centroid_lat'].where(ar_test['centroid_lat'] != -999)

    # Manually shift longitude values to center the region of interest
    lon_values = ar_test['centroid_lon'].values - 180
    lat_values = ar_test['centroid_lat'].values

    if np.max(lon_values) > 180:
        # Split trajectory into two parts
        split_index = np.argmax(lon_values > 180)
        lon_values_part1 = lon_values[:split_index]
        lat_values_part1 = lat_values[:split_index]

        # Switch coordinates to -180 to 180 for part2
        lon_values_part2 = lon_values[split_index:] - 360
        lat_values_part2 = lat_values[split_index:]

        # Plot part1
        ax.plot(lon_values_part1, lat_values_part1, alpha=0.3, color='gray', zorder=2)

        # Plot part2
        ax.plot(lon_values_part2, lat_values_part2, alpha=0.3, color='gray', zorder=2)

    else:
        # Plot the trajectory directly
        ax.plot(lon_values, lat_values, alpha=0.3, color='gray', zorder=2)

    # Plot the start and end points
    ax.scatter(lon_values[-1], lat_values[-1], alpha=0.2, color='red', zorder=3, label='End')
    ax.scatter(lon_values[0], lat_values[0], alpha=0.2, color='blue', zorder=3, label='Start')
    # plt.legend()  

# plt.legend()    

# Set the desired extent manually
ax.set_xlim([-110, 80])
ax.set_ylim([-15, 75])


# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5)
gl.ypadding = 5
gl.xpadding = 5
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 45)[::1])
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 45)[::1])

fig.text(0.50, 0.02, 'This composite constitutes '+num_ars+' AR systems at hourly resolution June 2000 through June 2024',
         horizontalalignment='center', wrap=True)
fig.savefig("/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/Figures/Trajectories/DJFM_MJO_Active_Trajectory_Spag.png", dpi=350, bbox_inches='tight') 
# fig.show()

## MJO-Connected

mjo_connected_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_MJO_Connected_ARs.csv')
mjo_connected_df = mjo_connected_df.drop(columns=['Unnamed: 0'])

dec=mjo_connected_df[mjo_connected_df['Month'] == 12]
jan=mjo_connected_df[mjo_connected_df['Month'] == 1]
feb=mjo_connected_df[mjo_connected_df['Month'] == 2]
mar=mjo_connected_df[mjo_connected_df['Month'] == 3]

CA_non_DJF=pd.concat([dec, jan, feb, mar], ignore_index=True)

Pac_ars_df = CA_non_DJF

num_ars = str(len(Pac_ars_df))
#plot and let's see
cm = 180
fig = plt.figure(figsize=[14, 7])
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
# Set central_longitude to 0
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title('DJFM Trajectory of MJO-Connected AR Systems', fontsize=14)
ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
ax.coastlines('50m', linewidth=2, zorder=5)

# Loop through ARs
for i in range(0,len(Pac_ars_df)):
    ar_test = xr.open_dataset(Pac_ars_df['AR ID (string)'].iloc[i])

    ar_test['centroid_lon'] = ar_test['centroid_lon'].where(ar_test['centroid_lon'] != -999)
    ar_test['centroid_lat'] = ar_test['centroid_lat'].where(ar_test['centroid_lat'] != -999)

    # Manually shift longitude values to center the region of interest
    lon_values = ar_test['centroid_lon'].values - 180
    lat_values = ar_test['centroid_lat'].values

    if np.max(lon_values) > 180:
        # Split trajectory into two parts
        split_index = np.argmax(lon_values > 180)
        lon_values_part1 = lon_values[:split_index]
        lat_values_part1 = lat_values[:split_index]

        # Switch coordinates to -180 to 180 for part2
        lon_values_part2 = lon_values[split_index:] - 360
        lat_values_part2 = lat_values[split_index:]

        # Plot part1
        ax.plot(lon_values_part1, lat_values_part1, alpha=0.3, color='gray', zorder=2)

        # Plot part2
        ax.plot(lon_values_part2, lat_values_part2, alpha=0.3, color='gray', zorder=2)

    else:
        # Plot the trajectory directly
        ax.plot(lon_values, lat_values, alpha=0.3, color='gray', zorder=2)

    # Plot the start and end points
    ax.scatter(lon_values[-1], lat_values[-1], alpha=0.2, color='red', zorder=3, label='End')
    ax.scatter(lon_values[0], lat_values[0], alpha=0.2, color='blue', zorder=3, label='Start')
    # plt.legend()  

# plt.legend()    

# Set the desired extent manually
ax.set_xlim([-110, 80])
ax.set_ylim([-15, 75])


# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5)
gl.ypadding = 5
gl.xpadding = 5
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 45)[::1])
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 45)[::1])

fig.text(0.50, 0.02, 'This composite constitutes '+num_ars+' AR systems at hourly resolution June 2000 through June 2024',
         horizontalalignment='center', wrap=True)
fig.savefig("/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/Figures/Trajectories/DJFM_MJO_Connected_Trajectory_Spag.png", dpi=350, bbox_inches='tight') 
# fig.show()

## Total ARs that makes landfall
CA_non_MAM=pd.concat([mjo_connected_df, mjo_active_df, non_mjo_df], ignore_index=True)

dec=CA_non_MAM[CA_non_MAM['Month'] == 12]
jan=CA_non_MAM[CA_non_MAM['Month'] == 1]
feb=CA_non_MAM[CA_non_MAM['Month'] == 2]
mar=CA_non_MAM[CA_non_MAM['Month'] == 3]

CA_non_DJF=pd.concat([dec, jan, feb, mar], ignore_index=True)

Pac_ars_df = CA_non_DJF

num_ars = str(len(Pac_ars_df))
#plot and let's see
cm = 180
fig = plt.figure(figsize=[14, 7])
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
# Set central_longitude to 0
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title('DJFM Trajectory of All Landfalling AR Systems', fontsize=14)
ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
ax.coastlines('50m', linewidth=2, zorder=5)

# Loop through ARs
for i in range(0,len(Pac_ars_df)):
    ar_test = xr.open_dataset(Pac_ars_df['AR ID (string)'].iloc[i])

    ar_test['centroid_lon'] = ar_test['centroid_lon'].where(ar_test['centroid_lon'] != -999)
    ar_test['centroid_lat'] = ar_test['centroid_lat'].where(ar_test['centroid_lat'] != -999)

    # Manually shift longitude values to center the region of interest
    lon_values = ar_test['centroid_lon'].values - 180
    lat_values = ar_test['centroid_lat'].values

    if np.max(lon_values) > 180:
        # Split trajectory into two parts
        split_index = np.argmax(lon_values > 180)
        lon_values_part1 = lon_values[:split_index]
        lat_values_part1 = lat_values[:split_index]

        # Switch coordinates to -180 to 180 for part2
        lon_values_part2 = lon_values[split_index:] - 360
        lat_values_part2 = lat_values[split_index:]

        # Plot part1
        ax.plot(lon_values_part1, lat_values_part1, alpha=0.3, color='gray', zorder=2)

        # Plot part2
        ax.plot(lon_values_part2, lat_values_part2, alpha=0.3, color='gray', zorder=2)

    else:
        # Plot the trajectory directly
        ax.plot(lon_values, lat_values, alpha=0.3, color='gray', zorder=2)

    # Plot the start and end points
    ax.scatter(lon_values[-1], lat_values[-1], alpha=0.2, color='red', zorder=3, label='End')
    ax.scatter(lon_values[0], lat_values[0], alpha=0.2, color='blue', zorder=3, label='Start')
    # plt.legend()  

# plt.legend()    

# Set the desired extent manually
ax.set_xlim([-110, 80])
ax.set_ylim([-15, 75])


# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5)
gl.ypadding = 5
gl.xpadding = 5
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 45)[::1])
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 45)[::1])

fig.text(0.50, 0.02, 'This composite constitutes '+num_ars+' AR systems at hourly resolution June 2000 through June 2024',
         horizontalalignment='center', wrap=True)
fig.savefig("/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/Figures/Trajectories/DJFM_All_LF_Trajectory_Spag.png", dpi=350, bbox_inches='tight') 
# fig.show()
