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
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import geopandas as gpd
# from shapely.geometry import mapping, Point, Polygon
# from shapely.ops import cascaded_union
# from osgeo import ogr
# from osgeo import gdal
# from osgeo import osr
# from osgeo import gdal_array
# from osgeo import gdalconst

## This script by Chad Small plots out the leading edge of ARs that make landfall on the West Coast of the U.S. that take place when the MJO is active, but not connected

#Bring the AR data

mjo_og = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/ERM_3day_Non_MJO_ARs.csv')
mjo_og = mjo_og.drop(columns=['Unnamed: 0'])

MJO_pac=mjo_og

#Bring the West Coast Boundary. Note: you'll have to swap out for individual state boundaries when you run this analysis again
#need the mask too
cali_test =xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/state_masks/WC_US_state_mask_ERA5_FULL.nc')
cali_mask = cali_test['Mask']
WC_bound = cali_mask

#loop through to get the leading edge data by finding all of AR points without our land boundary and taking the easternmost point

#add second loop for the lifetime of the AR
AR_ID = []
AR_MJO_match = []
AR_MJO_DT = []
start_lats = []
start_lons = []
end_lats = []
end_lons = []

max_ar_erm = []
AR_ID = []
AR_date = []
# erm_mask_fin = []
erm_masks = []
AR_ID_num = []
AR_month = []
AR_IVT = []
end_seas_yr = []

for i in range(0,len(MJO_pac)):
# for i in range(0,5):
    # url = i
    ar_test = xr.open_dataset(MJO_pac['AR ID (string)'].iloc[i])
    mask_ary=ar_test['mask'][0]

    for j in range(0, len(ar_test['mask']['time'])):
        mask_ary=ar_test['mask'][j]
        
        #pull the datetime
        times = np.array(mask_ary['time'])
        times=np.array2string(times)
        str_times = times.replace("'", "")
        fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')


        #lets' try adding the data frames back to back
        test_add = WC_bound + mask_ary
        htest=np.array((test_add == 2).any())
        bool_match=np.array(htest, dtype=bool)

        AR_ID += [MJO_pac['AR ID (string)'].iloc[i]]
        AR_MJO_match += [np.array((test_add == 2).any()).astype(str)]
        AR_MJO_DT += [fin_time.strftime('%Y-%m-%dT%H:%M:%S')]
        # AR_ID += [MJO_pac['AR ID (string)'].iloc[i]]
        AR_date += [MJO_pac['Landfall Datetime'].iloc[i]]
        AR_ID_num += [MJO_pac['AR ID (Number)'].iloc[i]]
        AR_month += [MJO_pac['Month'].iloc[i]]
        AR_IVT += [MJO_pac['Max IVT'].iloc[i]]
        end_seas_yr += [MJO_pac['End Season Year'].iloc[i]]
        max_ar_erm += [MJO_pac['Max ERM'].iloc[i]]

        ar_oi=xr.open_dataset(MJO_pac['AR ID (string)'].iloc[i])

        lon_2d, lat_2d = np.meshgrid(ar_oi['lon'].data,ar_oi['lat'].data)
        max_lon = []
        max_lat = []

        lon_in_mask = lon_2d[mask_ary==1]
        lat_in_mask = lat_2d[mask_ary==1]

        if len(lon_2d[mask_ary==1]) == 0:
            max_lon.append(np.nan)
            max_lat+=[np.nan]

        else:
            mx_lon=np.nanmax(lon_2d[mask_ary==1])
            # print(jjj)

            max_lon.append(mx_lon)
            max_lat+=[lat_in_mask[np.argmax(lon_in_mask)]]
        
        #make arrays and drop nans
        max_lat = np.array(max_lat)
        max_lon = np.array(max_lon)


        # lon_values = max_lon - 180
        lon_values = max_lon
        lat_values = max_lat


        end_lats += [lat_values[-1]]
        end_lons += [lon_values[-1]]

        #let's try to do the plot inside a if else statements
        if bool_match == True:
            NNNN=18

        else: 
            print("AR doesn't make landfall")

print('Create Dataframe for Output.')
# test_overall_df = pd.DataFrame({'AR ID (string)':AR_ID, 
#                                'Landfall?':AR_MJO_match,
#                                'Date':AR_MJO_DT,
#                                'End Lat':end_lats,
#                                'End Lon':end_lons})


test_overall_df = pd.DataFrame({'AR ID (string)':AR_ID, 
                               'Landfall Datetime':AR_date,
                               'Max ERM':max_ar_erm,
                               'Max IVT':AR_IVT,
                               'End Season Year':end_seas_yr,
                               'Month':AR_month,
                               'AR ID (Number)':AR_ID_num,
                               'Landfall?':AR_MJO_match,
                               'End Lat':end_lats,
                               'End Lon':end_lons})

#start pairing down to find my analysis window

#filter out what's true and then take the first instance of the highest lon per AR ID here
LF_df=test_overall_df.loc[test_overall_df['Landfall?'] == 'True'].reset_index(drop=True)
LF_df = LF_df.sort_values(by="End Lon").reset_index(drop=True)
fin_df = LF_df.drop_duplicates(["AR ID (string)"], keep="last").reset_index(drop=True)

# I should save this for future use
fin_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/LE_Non_MJO.csv') 

