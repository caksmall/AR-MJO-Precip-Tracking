from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset
import matplotlib
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors
import matplotlib.colors as colors
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
import os

## This script by Chad Small takes the Split/Merged AR systems that have an AR-MJO LPT Connection over some portion of their familial lifecycle and determines whether the AR-MJO connection precedes/coincides with their start time

#bring in the AR family and split merge data
#bring AR-MJO Families
mjo_conn_ovr_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Connected_AR_Fams.csv')
mjo_conn_ovr_df = mjo_conn_ovr_df.drop(columns=['Unnamed: 0'])

#bring in all split/merged ARs
split_mrg_mjo_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Connected_AR_Split_Merge_Raw.csv')
split_mrg_mjo_df = split_mrg_mjo_df.drop(columns=['Unnamed: 0'])

#let's put this all together into one loop
mjo_conn_ars = []
unclear_ars = []

for i in range(0,len(mjo_conn_ovr_df)):

# for i in range(0,5):

    #pull the string for the year
    MMMMMM = 59
    nn_len = 0
    year_str=str(mjo_conn_ovr_df['AR ID'].iloc[i])[nn_len - MMMMMM:-38]

    #pull through big data frame with this string to get the right year
    filtered_df = split_mrg_mjo_df[split_mrg_mjo_df['AR ID (string)'].str.contains(year_str, case=False, na=False)].reset_index(drop=True)

    #try to make a loop or something that filters for just the lpt id
    NNNN=18
    t_len=0
    lpt_id_tit=str(mjo_conn_ovr_df['AR ID'].iloc[i])[t_len - NNNN:-8]

    #output the times of interest

    #set up a second filter for the lpt id from the familiy file
    filtered_2pt_df = filtered_df[filtered_df['AR ID (string)'].str.contains(lpt_id_tit, case=False, na=False)].reset_index(drop=True)
    
    
    #let's get the date out of the AR-MJO familiy one
    ar_fam_dt=np.datetime64(mjo_conn_ovr_df['Overlap Datetimes'].iloc[i])

    for j in range(0,len(filtered_2pt_df)):

        #let's open up our AR split/merge files
        ar_oi_ds=xr.open_dataset(filtered_2pt_df['AR ID (string)'].iloc[j])

        #get the start and end dates
        ar_oi_start_dt=ar_oi_ds['time'][0].values
        # ar_oi_end_dt=ar_oi_ds['time'][-1].values

        if ar_fam_dt <= ar_oi_start_dt:
            print("AR system starts after AR-MJO Connection: It is MJO-Connected")
            mjo_conn_ars.append(filtered_2pt_df.loc[[j]])
        else:
            print("AR system starts BEFORE AR-MJO Connection: Not definitively MJO-Connected")
            unclear_ars.append(filtered_2pt_df.loc[[j]])

        

#concatenate outputs

mjo_final_df = pd.concat(mjo_conn_ars, ignore_index=True)
unclear_final_df = pd.concat(unclear_ars, ignore_index=True)   

#save outputs
mjo_final_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Connected_AR_Split_Merge_DEF.csv')
unclear_final_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Connected_AR_Split_Merge_UNCL.csv')