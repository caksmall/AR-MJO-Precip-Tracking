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

## This script by Chad Small counts the number of hours for MJO LPTs that are associated with ARS


#bring in MJO Impacted ARs
mjo_act_og = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_MJO_Active_ARs.csv')
mjo_act_og = mjo_act_og.drop(columns=['Unnamed: 0'])

mjo_og = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_MJO_Connected_ARs.csv')
mjo_og = mjo_og.drop(columns=['Unnamed: 0'])

mjo_overall_ars = pd.concat([mjo_act_og,mjo_og], ignore_index=True)

dec=mjo_overall_ars[mjo_overall_ars['Month'] == 12]
jan=mjo_overall_ars[mjo_overall_ars['Month'] == 1]
feb=mjo_overall_ars[mjo_overall_ars['Month'] == 2]
mar=mjo_overall_ars[mjo_overall_ars['Month'] == 3]

CA_non_DJF=pd.concat([dec, jan, feb, mar], ignore_index=True)

mjo_djfm = CA_non_DJF

#find the times
strm_times = []
for j in range(0,len(mjo_djfm)):
# for j in range(0,20):
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


mjo_centroid_list = []
for jj in clean_strm:
    year = jj.year
    month = jj.month

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


mjo_times = []

for iii in mjo_centroid_list:

    mjo_centroid_ds = xr.open_dataset(iii)
    
    

    for i in range(0,len(mjo_centroid_ds['time'])):
    # for i in range(0,5):
        times = np.array(mjo_centroid_ds['time'][i])
        times=np.array2string(times)
        str_times = times.replace("'", "")
        # str_times = str_times.replace("[", "")
        fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')

        if fin_time.month in [12,1,2,3]: #just the times within DJFM 

            mjo_times += [fin_time]

mjo_times = np.array(mjo_times)
fin_mjo_times_ls = np.unique(mjo_times)     

print(len(fin_mjo_times_ls))