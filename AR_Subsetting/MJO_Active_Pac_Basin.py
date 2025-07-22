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

## This script by Chad Small finds AR split/merge systems that took place an active MJO was present at some point during the lifetime but did not connect to MJO LPT

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

#bring in the Non-MJO and MJO-Connected ARs

mjo_conn_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Connected_ARs.csv')
mjo_conn_df = mjo_conn_df.drop(columns=['Unnamed: 0'])

non_mjo_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_Non_MJO_ARs.csv')
non_mjo_df = non_mjo_df.drop(columns=['Unnamed: 0'])


combined_df = pd.concat([mjo_conn_df, non_mjo_df]).reset_index(drop=True)

#now drop the ID's in this one that match in the other one
no_mjo_list=final_df[~final_df['AR ID (string)'].isin(combined_df['AR ID (string)'])].reset_index(drop=True)

# shouldn't be duplicates, but better safe than sorry
no_dup_no_mjo =no_mjo_list.drop_duplicates(subset='AR ID (string)',keep='first').reset_index(drop=True)
print(len(no_dup_no_mjo))

print('saving csv')
#save outputs
no_dup_no_mjo.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Active_ARs.csv')