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
from collections import Counter
from scipy.stats import mannwhitneyu

## This script by Chad Small plots the DJFM MJO LPT spatial distribution with the MJO LPT trajectories

#plot the whole thing
fl_names = glob.glob('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/nc_files/MJO_LPT_NC/DJFM_LPT_Hours_Clim_pt_*.nc')

ls_masks = []
for i in fl_names:
    ar_oi = xr.open_dataset(i)
    test1=np.array(ar_oi['MJO LPT Masks'])
    ls_masks += [test1]

plt_sum=sum(ls_masks)

#I also need the coordinates
ar_oi = xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/nc_files/MJO_LPT_NC/DJFM_LPT_Hours_Clim_pt_58.nc')
lon_ary = ar_oi['lon']
lat_ary = ar_oi['lat']

norm_val = 69801
plt_per = (plt_sum/norm_val)*100

#get the MJO LPT systems and their trajectories


#Dec
month_to_check = 12
dec_dt = []

for i in range(2000,2024):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            dec_dt += [dt.datetime(i,month_to_check,j,k)]

#Jan
month_to_check = 1
jan_dt = []

for i in range(2001,2025):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            jan_dt += [dt.datetime(i,month_to_check,j,k)]

#Feb
month_to_check = 2
feb_dt = []

for i in range(2001,2025):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            feb_dt += [dt.datetime(i,month_to_check,j,k)]

#Mar
month_to_check = 3
mar_dt = []

for i in range(2001,2025):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            mar_dt += [dt.datetime(i,month_to_check,j,k)]


#Apr
month_to_check = 4
apr_dt = []

for i in range(2001,2025):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            apr_dt += [dt.datetime(i,month_to_check,j,k)]

#May
month_to_check = 5
may_dt = []

for i in range(2001,2025):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            may_dt += [dt.datetime(i,month_to_check,j,k)]  

#Jun
month_to_check = 6
jun_dt = []

for i in range(2000,2025):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            jun_dt += [dt.datetime(i,month_to_check,j,k)]           

#jul
month_to_check = 7
jul_dt = []

for i in range(2000,2024):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            jul_dt += [dt.datetime(i,month_to_check,j,k)]             


#aug
month_to_check = 8
aug_dt = []

for i in range(2000,2024):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            aug_dt += [dt.datetime(i,month_to_check,j,k)]  

#sep
month_to_check = 9
sep_dt = []

for i in range(2000,2024):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            sep_dt += [dt.datetime(i,month_to_check,j,k)]  


#oct
month_to_check = 10
oct_dt = []

for i in range(2000,2024):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            oct_dt += [dt.datetime(i,month_to_check,j,k)]  


#nov
month_to_check = 11
nov_dt = []

for i in range(2000,2024):
# for i in range(1999,2004):
    month_to_itr = month_to_check
    # yr_clim = i
    ldom_clim = calendar.monthrange(i, month_to_itr)[1]
    for j in range(1,ldom_clim+1):
        # day_clim = j
        for k in range(0,24):
        # hour_clim = k

            nov_dt += [dt.datetime(i,month_to_check,j,k)]  

strm_times = dec_dt + jan_dt +feb_dt + mar_dt 

strm_times = np.array(strm_times)
clean_strm = np.unique(strm_times)

# len_oi = len(clean_strm)
# num_break = 35
# strt_range = int((len_oi/num_break)*bef_ints)
# end_range = int((len_oi/num_break)*instance)

mjo_centroid_list = []
for jj in range(0,len(clean_strm)):
    year = clean_strm[jj].year
    month = clean_strm[jj].month

    if month in [1,2,3,4,5]:
        year_new = year - 1
        nxt_year = year_new + 1

        #create a list of mjo files
        mjo_centroid_ls_1 = sorted(glob.glob('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/'+str(year_new)+'060100_'+str(nxt_year)+'063023/*.nc'))
        mjo_centroid_list += [mjo_centroid_ls_1]

    else:
        # year = year - 1 #turn off for everything not at the end! so for 1 through 199 turn this off

        year_new = year
        nxt_year = year + 1

        #create a list of mjo files
        mjo_centroid_ls_2 = sorted(glob.glob('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/'+str(year_new)+'060100_'+str(nxt_year)+'063023/*.nc'))
        mjo_centroid_list += [mjo_centroid_ls_2]

# Flatten the list of lists
mjo_centroid_list = [item for sublist in mjo_centroid_list for item in sublist]

mjo_centroid_list = list(set(mjo_centroid_list))

print(len(mjo_centroid_list))

#set up a list of arrays such that we can easily reference them for purpose of plotting

list_mjo_lons = []
list_mjo_lats = []

for ii in mjo_centroid_list:
    mjo_ds = xr.open_dataset(ii)
    mjo_cen_lons = mjo_ds['centroid_lon'].values
    mjo_cen_lats = mjo_ds['centroid_lat'].values

    mjo_cen_lons = np.where(mjo_cen_lons == -999, np.nan, mjo_cen_lons)

    # Drop NaNs (e.g., flatten then drop)
    mjo_cen_lons_clean = mjo_cen_lons[~np.isnan(mjo_cen_lons)]

    mjo_cen_lats = np.where(mjo_cen_lats == -999, np.nan, mjo_cen_lats)

    # Drop NaNs (e.g., flatten then drop)
    mjo_cen_lats_clean = mjo_cen_lats[~np.isnan(mjo_cen_lats)]

    list_mjo_lons += [mjo_cen_lons_clean[0:-1:12]] #plot every 12 hours
    list_mjo_lats += [mjo_cen_lats_clean[0:-1:12]] #plot every 12 hours

#plot or try to 
#plot and let's see


cm = 180
colormap=cmaps.WhiteBlueGreenYellowRed
fig = plt.figure(figsize=[14, 7])
political_boundaries = NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
states = NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='none')

ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
ax.set_title('DJFM MJO LPT System Frequency 24-Year Composite', fontsize=14)					   
ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
ax.coastlines('50m', linewidth=2, zorder=5)
minlon = 150 + cm
maxlon = -130 + cm

ax.set_extent([minlon, maxlon, -30, 75], ccrs.PlateCarree())
# css = ax.pcolormesh(lon_ary, lat_ary, plt_per, cmap = colormap, transform=ccrs.PlateCarree(),zorder=1, vmin=0,vmax=67.10720806886357, alpha=.1)


# Loop through ARs
for i in range(0,len(list_mjo_lons)):

    # Manually shift longitude values to center the region of interest
    lon_values = list_mjo_lons[i] - 180
    lat_values = list_mjo_lats[i]

    if np.max(lon_values) > 180:
        # Split trajectory into two parts
        split_index = np.argmax(lon_values > 180)
        lon_values_part1 = lon_values[:split_index]
        lat_values_part1 = lat_values[:split_index]

        # Switch coordinates to -180 to 180 for part2
        lon_values_part2 = lon_values[split_index:] - 360
        lat_values_part2 = lat_values[split_index:]

        # Plot part1
        ax.plot(lon_values_part1, lat_values_part1, alpha=0.5, color='gray', zorder=10)

        # Plot part2
        ax.plot(lon_values_part2, lat_values_part2, alpha=0.5, color='gray', zorder=10)

    else:
        # Plot the trajectory directly
        ax.plot(lon_values, lat_values, alpha=0.3, color='gray', zorder=10)

    # Plot the start and end points
    srt_p = ax.scatter(lon_values[-1], lat_values[-1], alpha=0.8, color='red', zorder=11, label='MJO End',marker='s')
    end_p = ax.scatter(lon_values[0], lat_values[0], alpha=0.8, color='blue', zorder=11, label='MJO Start',marker='d')

# cbar = plt.colorbar(css, ax=ax,orientation='horizontal',aspect=30, shrink=0.7, pad=0.06)
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Percent of MJO LPT Hours', fontsize =16)

legend=plt.legend(handles=[srt_p,end_p], loc='upper left')
legend.set_zorder(10)

#try adding gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5)
gl.ypadding = 5
gl.xpadding = 5
gl.xlocator = mticker.FixedLocator(np.arange(-180,180,45)[::1])
gl.ylocator = mticker.FixedLocator(np.arange(-90,90,45)[::1])
fig.text(0.50, 0.02,'This composite constitutes '+ str(len(mjo_centroid_list))+' MJO LPT Systems from June 2000 through June 2024', horizontalalignment='center', wrap=True )		
fig.savefig("/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/Figures/MJO_LPT_Distribution/DJFM_MJO_LPT_Distribution_Climatology_Traj.png", dpi=350, bbox_inches='tight') 
# fig.show()



    
    