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

#This script, by Chad Small, finds all of the AR Families (i.e., the full split and merge origin tracks) that terminate within the northern Pacific Basin region of analysis below

ls_contents = []
for i in range(2000,2024):
    start_year = i
    end_year = i + 1
    listing = sorted(glob.glob('/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/lpt-python-public/ar.testing7.merge_split.families/data/ar/g0_0h/thresh1/systems/'+str(i)+'060100_'+str(end_year)+'063023/*.nc'))
    ls_contents += [listing]

ls_contents = np.array(ls_contents)
ls_contents = np.concatenate(ls_contents)



ar_id = []
ar_year = []

for jj in ls_contents:
    # for i in range(1000, 1200):
    ar_oi = xr.open_dataset(jj)
    fin_time = len(ar_oi['time']) - 1
    ar_last=ar_oi['mask'][fin_time]
    LatIndexer, LonIndexer = 'lat', 'lon'
    slice_ar= ar_last.sel(**{LatIndexer: slice(60, 0),
                            LonIndexer: slice(100, 250)})
    if slice_ar.max() == 0:
            print('AR system not in domain')
    else:
        ar_year += [ar_oi['time'][-1].values.astype('datetime64[Y]').astype(int) + 1970]
        ar_id += [str(jj)]
        

AR_PAC_df= pd.DataFrame({'AR ID':ar_id, 
                            'End Season Year':ar_year})
    
AR_PAC_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Total_Pac_Basin_AR_Fams.csv')