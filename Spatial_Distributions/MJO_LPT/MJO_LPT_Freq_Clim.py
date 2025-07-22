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
from scipy.stats import mannwhitneyu
import sys

## This script by Chad Small calculates density of MJO LPT masks by LPT hours based of all MJO LPTs in the North Pacific Basin for DJFM.

instance = int(sys.argv[1])
bef_ints = instance - 1

#get the datetimes for all times in these months

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

strm_times = dec_dt + jan_dt +feb_dt + mar_dt + apr_dt + may_dt + jun_dt + jul_dt + aug_dt + sep_dt + oct_dt + nov_dt

strm_times = np.array(strm_times)
clean_strm = np.unique(strm_times)

len_oi = len(clean_strm)
# Need to break up this list for ease of use

num_break = 200
strt_range = int((len_oi/num_break)*bef_ints)
end_range = int((len_oi/num_break)*instance)

mjo_masks = []
for jj in range(strt_range,end_range):
# for jj in clean_strm:
# for jj in clean_strm[0:200]:

    year = clean_strm[jj].year
    month = clean_strm[jj].month

    # year = jj.year
    # month = jj.month

    #need if else loop to deal with the fact that the year could be wrong timeline for the mjo stuff

    if month in [1,2,3,4,5]:
        year_new = year - 1
        nxt_year = year_new + 1

        #open up mjo data
        mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year_new)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
        mjo_mask=mjo_test['mask_with_filter_and_accumulation']

    else:
        year = year - 1 #turn off for everything not at the end! so for 1 through 199 turn this off


        nxt_year = year + 1


        #open up mjo data
        mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
        mjo_mask=mjo_test['mask_with_filter_and_accumulation'] #pull this out the loop

    LatIndexer, LonIndexer, TimeIndexer  = 'lat', 'lon', 'time'
    mjo_per_ar = mjo_mask.sel(**{TimeIndexer: slice(clean_strm[jj],clean_strm[jj])})

    mjo_masks += [mjo_per_ar]

# mjo_masks = []
# for i in range(0,len(mjo_per_ar['time'])):
#     mjo_masks += [mjo_per_ar[i]]

mjo_bulk=xr.concat(mjo_masks, dim="time")
mjo_sums=mjo_bulk.sum(dim='time', skipna=True)

norm_val=len(mjo_bulk['time'])

print(norm_val)

if isinstance(mjo_sums, xr.DataArray):
    lon_ary = mjo_sums.coords['lon'].values  # Extract the lon values from the xarray DataArray
elif isinstance(mjo_sums, np.ndarray):
    print("mjo_sums is already a NumPy array. Unable to access 'lon' coordinates.")
else:
    print("Unexpected type:", type(mjo_sums))


#save the p-value as netcdf for later use and figuring out how to make this shit work
nc_save=mjo_sums
# lon_ary = mjo_sums['lon']
# lat_ary = mjo_sums['lat']
# time_ary = np.array(ar_bulk['time'])
ny, nx = (mjo_sums.shape[0], mjo_sums.shape[1])
ncout = Dataset('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/nc_files/MJO_LPT_NC/LPT_Hours_Clim_pt_'+str(instance)+'.nc','w','NETCDF4'); # using netCDF3 for output format 
ncout.createDimension('lon',nx);
ncout.createDimension('lat',ny);
# ncout.createDimension('time',nyears);
lonvar = ncout.createVariable('lon','float64',('lon'));lonvar[:] = mjo_sums['lon'];
latvar = ncout.createVariable('lat','float64',('lat'));latvar[:] = mjo_sums['lat'];
#timevar = ncout.createVariable('time','float64',('time'));timevar.setncattr('units',unout);timevar[:]=date2num(datesout,unout);
myvar = ncout.createVariable('MJO LPT Masks','float64',('lat','lon'));myvar.setncattr('units','masks');myvar[:] = nc_save;
ncout.close();