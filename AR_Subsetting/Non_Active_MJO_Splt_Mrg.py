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
import os
import sys

## This script by Chad Small finds AR split/merge systems that took place when no active MJO was present over the duration of their lifetimes only for the Pacific Basin

#bring in all Pac Basin AR Familes
pac_bas_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Total_Pac_Basin_AR_Fams.csv')
pac_bas_df = pac_bas_df.drop(columns=['Unnamed: 0'])

#bring in all split/merged ARs
split_mrg_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Total_Global_Splt_MRG_ARs.csv')
split_mrg_df = split_mrg_df.drop(columns=['Unnamed: 0'])

#bring in all the AR split/merge systems
saved_dfs = []

for i in range(0,len(pac_bas_df)):

# for i in range(20,30):

    #pull the string for the year
    MMMMMM = 59
    nn_len = 0
    year_str=str(pac_bas_df['AR ID'].iloc[i])[nn_len - MMMMMM:-38]

    #pull through big data frame with this string to get the right year
    filtered_df = split_mrg_df[split_mrg_df['AR ID (string)'].str.contains(year_str, case=False, na=False)].reset_index(drop=True)

    #try to make a loop or something that filters for just the lpt id
    NNNN=18
    t_len=0
    lpt_id_tit=str(pac_bas_df['AR ID'].iloc[i])[t_len - NNNN:-8]

    #output the times of interest

    #set up a second filter for the lpt id from the familiy file
    filtered_2pt_df = filtered_df[filtered_df['AR ID (string)'].str.contains(lpt_id_tit, case=False, na=False)].reset_index(drop=True)


    # Append the filtered DataFrame to the list
    saved_dfs.append(filtered_2pt_df)

# Concatenate all filtered DataFrames
final_df = pd.concat(saved_dfs, ignore_index=True)
#this gives me all AR split/merged in the Pac Basin


year = int(sys.argv[1])
nxt_year = year + 1
timestring = str(year)+'060100_'+str(nxt_year)+'063023'

new_pac_df_ars = final_df[final_df['AR ID (string)'].str.contains(timestring, case=False, na=False)].reset_index(drop=True)


mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+timestring+'_mjo_lpt.nc')
mjo_mask=mjo_test['mask_with_filter_and_accumulation'] #pull this out the loop


AR_ID_new = []
AR_flag = []
AR_dt = []


# AR_ID = []
# AR_MJO_match = []
# AR_MJO_DT = []
AR_ivt = []
AR_ID_num = []
# AR_year_end = []

for i in range(0,len(new_pac_df_ars)):
# for i in range(0,3):
    ar_test = xr.open_dataset(new_pac_df_ars['AR ID (string)'].iloc[i])

    for j in range(0, len(ar_test['mask']['time'])):
        mask_ary=ar_test['mask'][j]

        
        #pull the datetime
        times = np.array(mask_ary['time'])
        times=np.array2string(times)
        str_times = times.replace("'", "")
        # start_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')
        fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')

        #open up mjo data
        # mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
        # mjo_mask=mjo_test['mask_with_accumulation'] #pull this out the loop

        TimeIndexer = 'time'
        mjo_slice = mjo_mask.sel(**{TimeIndexer: slice(fin_time, fin_time)})
        #lets' try adding the data frames back to back
        if len(mjo_slice) == 0:
            print("not in MJO date range")
        else:
            if mjo_slice[0].max() == 0:
                AR_ID_new += [new_pac_df_ars['AR ID (string)'].iloc[i]]
                AR_flag += [0]
                AR_dt += [new_pac_df_ars['End Season Year'].iloc[i]]
                AR_ivt += [new_pac_df_ars['Max IVT'].iloc[i]]
                AR_ID_num += [new_pac_df_ars['AR ID (Number)'].iloc[i]]
            else:
                print('Active MJO somewhere')
                AR_ID_new += [new_pac_df_ars['AR ID (string)'].iloc[i]]
                AR_flag += [1]
                AR_dt += [new_pac_df_ars['End Season Year'].iloc[i]]
                AR_ivt += [new_pac_df_ars['Max IVT'].iloc[i]]
                AR_ID_num += [new_pac_df_ars['AR ID (Number)'].iloc[i]]
                


no_active_mjo_df = pd.DataFrame({'AR ID (string)':AR_ID_new,
                                'MJO Flag':AR_flag,
                                'End Season Year':AR_dt,
                                'Max IVT':AR_ivt,
                                'AR ID (Number)':AR_ID_num})

MJO_act=no_active_mjo_df.loc[no_active_mjo_df['MJO Flag'] == 1].reset_index(drop=True)
MJO_non_act=no_active_mjo_df.loc[no_active_mjo_df['MJO Flag'] == 0].reset_index(drop=True)

MJO_act =MJO_act.drop_duplicates(subset='AR ID (string)',keep='first').reset_index(drop=True)
MJO_non_act =MJO_non_act.drop_duplicates(subset='AR ID (string)',keep='first').reset_index(drop=True)

clean_no_mjo=MJO_non_act[~MJO_non_act['AR ID (string)'].isin(MJO_act['AR ID (string)'])]

clean_no_mjo.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Non_Active_MJO_Pac_Basin_ARs_'+str(year)+'_'+str(nxt_year)+'.csv')