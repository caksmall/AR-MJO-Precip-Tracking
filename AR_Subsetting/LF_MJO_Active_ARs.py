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

#This script by Chad Small checks to see if the AR systems below make landfall anywhere on WC
#original script here
# /home/disk/orca/csmall3/AR_testing_research/src/Landfall_tests/Find_LF_WA_Non_mjo.py

#bring in the AR family data for ARs in the Pacific Basin
pac_ars_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Pac_Basin_MJO_Active_ARs.csv')
pac_ars_df = pac_ars_df.drop(columns=['Unnamed: 0'])

#need the mask too
cali_test =xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/state_masks/WC_US_state_mask_ERA5_FULL.nc')
cali_mask = cali_test['Mask']

mjo_mask= cali_mask

#set up loop to bring in AR times

AR_ID_txt = []
AR_ID_num = []
max_ivt = []
AR_MJO_match = []
AR_MJO_DT = []
AR_Seas_year = []

for jj in range(0,len(pac_ars_df)):
    #get title of AR ID string
    NNNN=18
    t_len = len(str(pac_ars_df['AR ID (string)'].iloc[jj]))
    lpt_id_tit=str(pac_ars_df['AR ID (string)'].iloc[jj])[t_len - NNNN:]

    #get the string for the MJO lpt times
    MMMMMM = 59
    nn_len = 0
    year_str=str(pac_ars_df['AR ID (string)'].iloc[jj])[nn_len - MMMMMM:-38]

    #open up mjo data
    # mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+year_str+'_mjo_lpt.nc')
    # mjo_mask=mjo_test['mask_with_filter_and_accumulation'] #pull this out the loop

    # set for loop here
    # AR_ID = []
    # AR_MJO_match = []
    # AR_MJO_DT = []

    # os.makedirs('/home/disk/orca/csmall3/AR_testing_research/src/Landfall_tests/Combined_MJO_src/Test_figs/tst_NEW_Deg_5_'+str(year)+'_'+str(nxt_year)+'_all_new_match/', exist_ok=True)

    #add second loop for the lifetime of the AR
    # for ni, i in enumerate(AR_ls):
        # url = i
    ar_test = xr.open_dataset(pac_ars_df['AR ID (string)'].iloc[jj])

        # print('{} of {} ARs.'.format(ni, len(AR_ls)))

    found_a_match = False

    for j in range(0, len(ar_test['time'])):
        # try:
        mask_ary=ar_test['mask'][j]
        
        #pull the datetime
        times = np.array(mask_ary['time'])
        times=np.array2string(times)
        str_times = times.replace("'", "")
        fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')

        #open up mjo data
        # mjo_test = xr.open_dataset('/home/orca/bkerns/lib/lpt/lpt-python-public/ERA5/data/era5/g20_72h/thresh12/systems/lpt_composite_mask_'+str(year)+'060100_'+str(nxt_year)+'063023_mjo_lpt.nc')
        # mjo_mask=mjo_test['mask_with_accumulation'] #pull this out the loop

        # TimeIndexer = 'time'
        # mjo_mask = mjo_mask.sel(**{TimeIndexer: slice(fin_time, fin_time)})

        #lets' try adding the data frames back to back
        # if len(mjo_mask) == 0:
        #     print("not in MJO date range")
        # else:
        test_add = mjo_mask + mask_ary
        htest=np.array((test_add == 2).any())
        bool_match=np.array(htest, dtype=bool)

        AR_ID_txt += [str(pac_ars_df['AR ID (string)'].iloc[jj])]
        AR_MJO_match += [np.array((test_add == 2).any()).astype(str)]
        max_ivt += [pac_ars_df['Max IVT'].iloc[jj]]
        AR_ID_num += [str(pac_ars_df['AR ID (Number)'].iloc[jj])]
        AR_Seas_year += [pac_ars_df['End Season Year'].iloc[jj]]
        AR_MJO_DT += [fin_time.strftime('%Y-%m-%dT%H:%M:%S')]  ### Remember you need the overlap time if apllicable!!! You can use this, just remember that it's going to be the first time in that LPT ID that's true

        #let's try to do the plot inside a if else statements
        if bool_match == True:
            # NNNN=18
            # t_len = len(str(url))
            # hrrr=str(url)[t_len - NNNN:]
            # TT =15
            # # str(hrrr)[:TT]
            # title = str(hrrr)[:TT]
            title= lpt_id_tit




            ## The plots 
            plot_data = mjo_mask
            lon_ary = mjo_mask['lon']
            lat_ary = mjo_mask['lat']

            plot_data2 = mask_ary
            lon_ary2 = mask_ary['lon']
            lat_ary2 = mask_ary['lat']

            colormap=cmaps.MPL_jet
            fig = plt.figure(figsize=[14, 7])
            political_boundaries = NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
            states = NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='none')

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title('MJO AR('+title+' '+ year_str +') Overlap '+str(fin_time), fontsize=14)					   
            ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
            ax.coastlines('50m', linewidth=2, zorder=5)
            ax.set_extent([190, 280, 0, 65], crs=ccrs.PlateCarree())#set west coast in a second
            css = ax.pcolormesh(lon_ary, lat_ary, plot_data, cmap = 'Greys', transform=ccrs.PlateCarree(),zorder=1, vmin=0, alpha=.5)
            css = ax.pcolormesh(lon_ary2, lat_ary2, plot_data2, cmap = 'Reds', transform=ccrs.PlateCarree(),zorder=1, vmin=0, alpha=.5)

            

            #try adding gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5)
            gl.ypadding = 5
            gl.xpadding = 5
            gl.xlocator = mticker.FixedLocator(np.arange(-180,180,45)[::1])
            gl.ylocator = mticker.FixedLocator(np.arange(-90,90,45)[::1])		
            # fig.show()
            plt.savefig('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/Figures/LF_MJO_Active_Match/'+lpt_id_tit+'_'+year_str+'.png')
            # plt.savefig('./figures/tst_NEW_Deg_0_'+str(year)+'_'+str(nxt_year)+'_all_new_match/'+title+'.png')
            plt.close(fig)

            found_a_match = True
            break
        else: 
            print("AR doesn't make landfall")
        
    if found_a_match:
        continue

        # except:
        #     pass

print('Create Dataframe for Output.')
AR_MJO_overlp_df = pd.DataFrame({'AR ID (string)':AR_ID_txt, 
                               'Overlap Datetimes':AR_MJO_DT,
                               'Landfall?':AR_MJO_match,
                               'Max IVT':max_ivt,
                               'End Season Year':AR_Seas_year,
                               'AR ID (Number)':AR_ID_num})

#to save the df later
print('Save Dataframe to CSV.')
# AR_MJO_overlp_df.to_csv('./Text_data/tst_NEW_Deg_0_Test_all_output_'+str(year)+'_'+str(nxt_year)+'.csv') 
AR_MJO_overlp_df.to_csv('/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/LF_MJO_Active_ARs.csv') 