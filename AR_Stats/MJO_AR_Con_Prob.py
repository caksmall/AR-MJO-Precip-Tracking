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

## This script by Chad Small tests the conditional probability of an AR exisiting given the MJO LPT is detected AND the probability of an AR exisiting given that there's no MJO LPT present


## make a loop/list of all times that can be run through for my data for both conditional probabilities
all_time_ranges = [
    (dt.datetime(2000, 6, 1), dt.datetime(2024, 6, 30))
]

# Initialize an empty list to store the result
all_times_list = []

# Loop through each range
for start, end in all_time_ranges:
    current = start
    while current <= end:
        # Append current datetime if it matches the 6-hour intervals
        if current.hour in range(0,24):
            all_times_list.append(current)
        # Increment by 6 hours
        current += dt.timedelta(hours=1)

all_times_list = np.array(all_times_list)

## AR Given MJO Cond Probability -----------------------------------------

mjo_centroid_list = []
for jj in all_times_list:
# for jj in all_times_list[0:10000]:
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

print('Hours MJO LPT is present in DJFM = '+str(len(fin_mjo_times_ls)))

## bring in the AR times when the MJO LPT is active/detected or whatever

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
        # strm_times += [fin_time]

        if fin_time.month in [12,1,2,3]: #just the times within DJFM 

            strm_times += [fin_time]

strm_times = np.array(strm_times)
clean_strm = np.unique(strm_times)

## find the hours where the AR is present and MJO LPT is present

overlap_time_ls = []
for i in range(0,len(clean_strm)):
# for i in range(0,400):
    if clean_strm[i] in fin_mjo_times_ls:
        overlap_time_ls += [clean_strm[i]]
    else:
        print('no overlap')


print('Hours AR AND MJO LPT are present in DJFM = '+str(len(overlap_time_ls)))

AR_MJO_cond_prob = len(overlap_time_ls)/len(fin_mjo_times_ls)

print('Conditional Probability of AR Given MJO = '+str(AR_MJO_cond_prob))


## AR Given NO MJO Cond Probability -----------------------------------------

#need to make sure these times are just in DJFM otherwise it's going to act up

djfm_all_times = []

for i in range(0,len(all_times_list)):
    fin_time_2 = all_times_list[i]

    if fin_time_2.month in [12,1,2,3]: #just the times within DJFM 

        djfm_all_times += [fin_time_2]

djfm_all_times = np.array(djfm_all_times)

non_mjo_times = []

for i in range(0, len(djfm_all_times)):
    if djfm_all_times[i] in fin_mjo_times_ls:
        print('this time is MJO; skip')

    else:
        non_mjo_times += [djfm_all_times[i]]

print('Hours MJO LPT is NOT present in DJFM = '+str(len(non_mjo_times)))

#bring in the non MJO ARs

#bring in MJO Impacted ARs
mjo_act_og = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_Non_MJO_ARs.csv')
mjo_act_og = mjo_act_og.drop(columns=['Unnamed: 0'])

mjo_overall_ars = mjo_act_og

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
        # strm_times += [fin_time]

        if fin_time.month in [12,1,2,3]: #just the times within DJFM 

            strm_times += [fin_time]

strm_times = np.array(strm_times)
clean_strm = np.unique(strm_times)

#try to make a loop to catch the times when ARs co occur with the MJO LPT times
overlap_time_ls_NON = []
for i in range(0,len(clean_strm)):
# for i in range(0,400):
    if clean_strm[i] in non_mjo_times:
        overlap_time_ls_NON += [clean_strm[i]]
    else:
        print('no overlap')

print('Hours AR is present AND NO MJO LPT is present in DJFM = '+str(len(overlap_time_ls_NON)))

AR_Non_MJO_cond_prob = len(overlap_time_ls_NON)/len(non_mjo_times)

print('Conditional Probability of AR Given No MJO = '+str(AR_Non_MJO_cond_prob))