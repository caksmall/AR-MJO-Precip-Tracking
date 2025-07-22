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

## This script by Chad Small extracts all MJO LPT volumetric rainfall values of every MJO LPT instance that is associated with MJO-Connected ARs (DJFM)
## In essence if an MJO LPT connects to an AR, this script pulls in the volumetric rainfall of every instance of that MJO LPT for the duration of the AR lifetime

#bring in the AR data
#try with mjo connected
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

print(len_oi)

mjo_rain = []
mjo_date = []
# for jj in range(strt_range,end_range):
for jj in range(0,len_oi):
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
        mjo_mask=mjo_test['volrain_with_filter_and_accumulation_tser']
        

    else:
        # year = year - 1 #turn off for everything not at the end! so for 1 through 199 turn this off


        nxt_year = year + 1


        #open up mjo data
        mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
        mjo_mask=mjo_test['volrain_with_filter_and_accumulation_tser'] #pull this out the loop

    LatIndexer, LonIndexer, TimeIndexer  = 'lat', 'lon', 'time'
    mjo_per_ar = mjo_mask.sel(**{TimeIndexer: slice(clean_strm[jj],clean_strm[jj])})
    # mjo_per_ar.values[0]

    mjo_rain += [mjo_per_ar.values[0]]
    mjo_date += [clean_strm[jj]]

mjo_rain_ary = np.array(mjo_rain)

#let's put this into a df so I can save it
print('Save Dataframe to CSV.')
mjo_vol_df = pd.DataFrame({'Date':mjo_date, 
                               'Volumetric Rain':mjo_rain_ary})
#save the ERM Data
mjo_vol_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/DJFM_MJO_LPT_Vol_Rain_MJO_Conn.csv') 
#this should take 6 minutes ish

