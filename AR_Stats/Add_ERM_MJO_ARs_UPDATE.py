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

## This script by Chad Small finds the max ERM for landfalling MJO-Connected ARs

## run this first
def find_majority_value(arr):
    n = len(arr)
    threshold = n // 2
    count = Counter(arr)
    
    for value, cnt in count.items():
        if cnt > threshold:
            return value
    
    return None  # Return None if no majority value found

#bring in test data for ARs
mjo_ars_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/LF_MJO_Connected_ARs.csv')
mjo_ars_df = mjo_ars_df.drop(columns=['Unnamed: 0'])


#add month data to ARs
ar_dt = []
for i in range(0,len(mjo_ars_df)):
    ar_test=xr.open_dataset(mjo_ars_df['AR ID (string)'].iloc[i])
    time_len = len(ar_test['time'])

    ls_months = []
    
    for j in range(0,time_len):
        ls_months += [ar_test['time'][j].values.astype('datetime64[M]').astype(int) % 12 + 1]
    
    ls_months = np.array(ls_months)

    ar_dt += [find_majority_value(ls_months)]

mjo_ars_df['Month'] = ar_dt

#bring in the West Coast Boundary
WC_IMERG_mask = xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/state_masks/US_West_Coast_mask_IMERG.nc')
# WC_IMERG_mask['lat'] = WC_IMERG_mask['lat'][::-1]
mask_ary = WC_IMERG_mask['Mask']
WC_bound = mask_ary

#this runs the ERM information
max_ar_erm = []
AR_ID = []
AR_date = []
# erm_mask_fin = []
erm_masks = []
AR_ID_num = []
AR_month = []
AR_IVT = []
end_seas_yr = []

for i in range(0,len(mjo_ars_df)):
# for i in range(0,1):
    times = np.array(mjo_ars_df['Overlap Datetimes'].iloc[i])
    times=np.array2string(times)
    str_times = times.replace("'", "")
    fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')#this gives you the datetime of first landfall
    fin_year = fin_time.year 

    if fin_year in [2000,2023,2024]: #2000 is set up different
        print('different time string for files!')
        ary_test = xr.open_dataset(mjo_ars_df['AR ID (string)'].iloc[i]) #open up the first AR system
        time_index = np.where(ary_test['time'] == np.datetime64(fin_time))[0][0]

        year_oi = []
        days_of_oi = []

        for iiii in range(-24,(len(ary_test['time']) - time_index)): #the solution is to read this into a df and then drop the duplicates!
            #-24 to give me the day before
            new_times = np.array2string(ary_test['time'][time_index + iiii].values)
            str_new_times = new_times.replace("'", "")
            fin_new_time = datetime.strptime(str_new_times.split(".")[0], '%Y-%m-%dT%H:%M:%S') #this gies you datetime of the subsequent time after landfall
            new_year = fin_new_time.year
            year_oi += [new_year]
            day_of_year = fin_new_time.timetuple().tm_yday
            days_of_oi += [day_of_year]

        hold_erm_dates_df = pd.DataFrame({'Year':year_oi, 
                            'Day of Year':days_of_oi})
        
        hold_erm_dates_df = hold_erm_dates_df.drop_duplicates().reset_index(drop=True)

        #create loop to read in the ERM data
        max_erm = []
        # erm_masks = []

        for yyyy in range(0,len(hold_erm_dates_df)):

            erm_test = xr.open_dataset('/home/orca/bkerns/projects/doe_coastal/extreme_rain/global_gridded_erm/data/processed/daily_precip_and_erm.3days.imerg_v7_2006_2020.'+str(hold_erm_dates_df['Year'].iloc[yyyy])+'.nc')
            erm_ary=erm_test['erm'][hold_erm_dates_df['Day of Year'].iloc[yyyy]-1] #minus 1 to deal with the indexing problems
            a,b=xr.align(erm_ary, mask_ary) #this works to fix the alignment problem!

            test_plt=a.where(b,0)#this works to mask perfectlY!

            #get the max erm
            erm_masks += [test_plt] #this gets a list of all the ERM masks when the ARs are over land

            max_erm += [test_plt.max().values] #this gives the spatial max, not for the whole systems

        
        max_erm = np.array(max_erm)
        max_ar_erm += [max_erm.max()] #this the max for the whole duration of the system
        AR_ID += [mjo_ars_df['AR ID (string)'].iloc[i]]
        AR_date += [mjo_ars_df['Overlap Datetimes'].iloc[i]]
        AR_ID_num += [mjo_ars_df['AR ID (Number)'].iloc[i]]
        AR_month += [mjo_ars_df['Month'].iloc[i]]
        AR_IVT += [mjo_ars_df['Max IVT'].iloc[i]]
        end_seas_yr += [mjo_ars_df['End Season Year'].iloc[i]]
    else:
        ary_test = xr.open_dataset(mjo_ars_df['AR ID (string)'].iloc[i]) #open up the first AR system
        time_index = np.where(ary_test['time'] == np.datetime64(fin_time))[0][0]

        year_oi = []
        days_of_oi = []

        for iiii in range(-24,(len(ary_test['time']) - time_index)): #the solution is to read this into a df and then drop the duplicates!
            #-24 to give me the day before
            new_times = np.array2string(ary_test['time'][time_index + iiii].values)
            str_new_times = new_times.replace("'", "")
            fin_new_time = datetime.strptime(str_new_times.split(".")[0], '%Y-%m-%dT%H:%M:%S') #this gies you datetime of the subsequent time after landfall
            new_year = fin_new_time.year
            year_oi += [new_year]
            day_of_year = fin_new_time.timetuple().tm_yday
            days_of_oi += [day_of_year]

        hold_erm_dates_df = pd.DataFrame({'Year':year_oi, 
                            'Day of Year':days_of_oi})
        
        hold_erm_dates_df = hold_erm_dates_df.drop_duplicates().reset_index(drop=True)

        #create loop to read in the ERM data
        max_erm = []
        # erm_masks = []

        for yyyy in range(0,len(hold_erm_dates_df)): #there's a possiblity that the year spans the time break and kicks it back into a different year by accident. Need to account for this by kicking back up to the other year

            if hold_erm_dates_df['Year'].iloc[yyyy] == 2023:
                erm_test = xr.open_dataset('/home/orca/bkerns/projects/doe_coastal/extreme_rain/global_gridded_erm/data/processed/daily_precip_and_erm.3days.imerg_v7_2006_2020.'+str(hold_erm_dates_df['Year'].iloc[yyyy])+'.nc')
                erm_ary=erm_test['erm'][hold_erm_dates_df['Day of Year'].iloc[yyyy]-1] #minus 1 to deal with the indexing problems
                a,b=xr.align(erm_ary, mask_ary) #this works to fix the alignment problem!

                test_plt=a.where(b,0)#this works to mask perfectlY!

                #get the max erm
                erm_masks += [test_plt] #this gets a list of all the ERM masks when the ARs are over land

                max_erm += [test_plt.max().values] #this gives the spatial max, not for the whole systems
            else:


                erm_test = xr.open_dataset('/home/orca/bkerns/projects/doe_coastal/extreme_rain/global_gridded_erm/data/processed/daily_precip_and_erm.3days.imerg_v7_2006_2020.'+str(hold_erm_dates_df['Year'].iloc[yyyy])+'.nc')
                erm_ary=erm_test['erm'][hold_erm_dates_df['Day of Year'].iloc[yyyy]-1] #minus 1 to deal with the indexing problems
                a,b=xr.align(erm_ary, mask_ary) #this works to fix the alignment problem!

                test_plt=a.where(b,0)#this works to mask perfectlY!

                #get the max erm
                erm_masks += [test_plt] #this gets a list of all the ERM masks when the ARs are over land

                max_erm += [test_plt.max().values] #this gives the spatial max, not for the whole systems

        
        max_erm = np.array(max_erm)
        max_ar_erm += [max_erm.max()] #this the max for the whole duration of the system
        AR_ID += [mjo_ars_df['AR ID (string)'].iloc[i]]
        AR_date += [mjo_ars_df['Overlap Datetimes'].iloc[i]]
        AR_ID_num += [mjo_ars_df['AR ID (Number)'].iloc[i]]
        AR_month += [mjo_ars_df['Month'].iloc[i]]
        AR_IVT += [mjo_ars_df['Max IVT'].iloc[i]]
        end_seas_yr += [mjo_ars_df['End Season Year'].iloc[i]]



# erm_bulk=xr.concat(erm_masks, dim="time")

print('Save Dataframe to CSV.')
ar_max_erm_df = pd.DataFrame({'AR ID (string)':AR_ID, 
                               'Landfall Datetime':AR_date,
                               'Max ERM':max_ar_erm,
                               'Max IVT':AR_IVT,
                               'End Season Year':end_seas_yr,
                               'Month':AR_month,
                               'AR ID (Number)':AR_ID_num})
#save the ERM Data
ar_max_erm_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_MJO_Connected_ARs_UPDATE.csv') 



