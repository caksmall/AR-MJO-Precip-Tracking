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
import sys

## This script (and the others with a similar name) brings in/creates the MJO LPT Hours for MJO-Connected ARs during a given RMM Phase
phase_oi = 4

rmm_nc=xr.open_dataset('/home/orca/data/indices/MJO/RMM/rmm.nc')

instance = int(sys.argv[1])
bef_ints = instance - 1

#bring in MJO-Connected ARs
mjo_og = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_MJO_Connected_ARs.csv')
mjo_og = mjo_og.drop(columns=['Unnamed: 0'])

dec=mjo_og[mjo_og['Month'] == 12]
jan=mjo_og[mjo_og['Month'] == 1]
feb=mjo_og[mjo_og['Month'] == 2]
mar=mjo_og[mjo_og['Month'] == 3]

CA_non_DJF=pd.concat([dec, jan, feb, mar], ignore_index=True)

mjo_djfm = CA_non_DJF
#find the times
strm_times = []
for j in range(0,len(mjo_djfm)):
    ar_oi = xr.open_dataset(str(mjo_djfm['AR ID (string)'].iloc[j]))
    for i in range(0,len(ar_oi['time'])):
        times = np.array(ar_oi['time'][i])
        times=np.array2string(times)
        str_times = times.replace("'", "")
        # str_times = str_times.replace("[", "")
        fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')
        strm_times += [fin_time]

strm_times = np.array(strm_times)
clean_strm = np.unique(strm_times)

len_oi = len(clean_strm)

num_break = 5
strt_range = int((len_oi/num_break)*bef_ints)
end_range = int((len_oi/num_break)*instance)

rmm_7_masks = []

for jj in range(strt_range,end_range):
# for jj in clean_strm:
# for jj in clean_strm[0:200]:

    day_cc = clean_strm[jj].day
    month_cc = clean_strm[jj].month
    year_cc = clean_strm[jj].year
    clean_strm_day = dt.datetime(year_cc,month_cc,day_cc)
    TimeIndexer = 'time'
    rmm_ds_oi = rmm_nc.sel(**{TimeIndexer: slice(clean_strm_day, clean_strm_day)})
    rmm_phase=rmm_ds_oi['phase'][0].data

    if rmm_phase == 4:
        
        year = clean_strm[jj].year
        month = clean_strm[jj].month

        if month in [1,2,3,4,5]:
            year_new = year - 1
            nxt_year = year_new + 1

            #open up mjo data
            mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year_new)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
            mjo_mask=mjo_test['mask_with_filter_and_accumulation']

        else:
            # year = year - 1 #turn off for everything not at the end! so for 1 through 199 turn this off


            nxt_year = year + 1


            #open up mjo data
            mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
            mjo_mask=mjo_test['mask_with_filter_and_accumulation'] #pull this out the loop

        LatIndexer, LonIndexer, TimeIndexer  = 'lat', 'lon', 'time'
        mjo_per_ar = mjo_mask.sel(**{TimeIndexer: slice(clean_strm[jj],clean_strm[jj])})

        rmm_7_masks += [mjo_per_ar]
    else:
        print('Not Phase '+str(phase_oi))



mjo_bulk=xr.concat(rmm_7_masks, dim="time")
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
ncout = Dataset('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/nc_files/MJO_LPT_NC/RMM_Phase/MJO_Connected/DJFM_MJO_Conn_LPT_Hours_RMM_4_pt_'+str(instance)+'.nc','w','NETCDF4'); # using netCDF3 for output format 
ncout.createDimension('lon',nx);
ncout.createDimension('lat',ny);
# ncout.createDimension('time',nyears);
lonvar = ncout.createVariable('lon','float64',('lon'));lonvar[:] = mjo_sums['lon'];
latvar = ncout.createVariable('lat','float64',('lat'));latvar[:] = mjo_sums['lat'];
#timevar = ncout.createVariable('time','float64',('time'));timevar.setncattr('units',unout);timevar[:]=date2num(datesout,unout);
myvar = ncout.createVariable('MJO LPT Masks','float64',('lat','lon'));myvar.setncattr('units','masks');myvar[:] = nc_save;
ncout.close();
