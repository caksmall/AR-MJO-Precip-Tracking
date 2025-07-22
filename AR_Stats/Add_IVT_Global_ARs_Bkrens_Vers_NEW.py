import numpy as np
import xarray as xr 
import datetime as dt 
import calendar
import cftime 
from multiprocessing import Pool
import tqdm
import sys
import pandas as pd
import glob 


def get_ivt_at_a_time_and_point(datetime_to_get, i_to_get, j_to_get):
    """
    get_ivt_at_a_time_and_point(datetime_to_get, i_to_get, j_to_get)

    Get the IVT value at a specified time and point(s).
    Specify multiple points as lists to get a list of IVT as output.
    """
    data_dir = '/home/orca/data/model_anal/era5/from_rda'

    year = datetime_to_get.year
    month = datetime_to_get.month
    eom_day = calendar.monthrange(year, month)[1] # Just the day.
    dt1 = dt.datetime(year, month, 1, 0, 0, 0)
    dt2 = dt.datetime(year, month, eom_day, 23, 0, 0)
    ymdh1 = dt1.strftime('%Y%m%d%H')
    ymdh2 = dt2.strftime('%Y%m%d%H')
    fn_viwve = f'{data_dir}/viwve/e5.oper.an.vinteg.162_071_viwve.ll025sc.{ymdh1}_{ymdh2}.nc'
    fn_viwvn = f'{data_dir}/viwvn/e5.oper.an.vinteg.162_072_viwvn.ll025sc.{ymdh1}_{ymdh2}.nc'

    with xr.open_dataset(fn_viwve, use_cftime=True) as ds_viwve0:
        ds_viwve = ds_viwve0.sel(time=datetime_to_get)
        viwve = ds_viwve['VIWVE'].data[j_to_get, i_to_get]#[0]

    with xr.open_dataset(fn_viwvn, use_cftime=True) as ds_viwvn0:
        ds_viwvn = ds_viwvn0.sel(time=datetime_to_get)
        viwvn = ds_viwvn['VIWVN'].data[j_to_get, i_to_get]#[0]

    ivt = np.sqrt(np.power(viwve,2) + np.power(viwvn,2))

    return ivt


def get_lpt_system_file_name(year1):

    year2 = year1 + 1

    data_dir = (
        '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/'
        + 'lpt-python-public/ar.testing/data/ar/g0_0h/thresh1/systems'
    )

    ymdh1 = f'{year1}060100'
    ymdh2 = f'{year2}063023'

    fn_lpt_systems = f'{data_dir}/lpt_systems_ar_{ymdh1}_{ymdh2}.nc'

    return fn_lpt_systems


def get_lptid_list(year1):

    fn_lpt_systems = get_lpt_system_file_name(year1)
    with xr.open_dataset(fn_lpt_systems) as ds:
        this_lptid_list = ds['lptid'].data

    return this_lptid_list

#let's try to get the literal string file name of the AR too
def get_lptid_name_list(year1):

    fn_lpt_systems = get_lpt_system_file_name(year1)
    # with xr.open_dataset(fn_lpt_systems) as ds:
    #     this_lptid_list = ds['lptid'].data

    return fn_lpt_systems


def get_lpo_list(year1, lptid):

    fn_lpt_systems = get_lpt_system_file_name(year1)

    with xr.open_dataset(fn_lpt_systems) as ds:
        this_lptid_idx = np.argwhere(ds['lptid'].data == lptid).flatten()[0]
        this_lpo_list = ds['objid'][this_lptid_idx, :].data
        this_lpo_list = this_lpo_list[np.isfinite(this_lpo_list)]
        this_lpo_list = [int(x) for x in this_lpo_list]

    return this_lpo_list


def get_points_from_lpo(lpo_id):

    data_dir_lpo = (
        '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/'
        + 'lpt-python-public/ar.testing/data/ar/g0_0h/thresh1/objects'
    )

    lpo_id = int(lpo_id)
    dt_this_str = str(int(lpo_id))[0:10]
    dt_this = dt.datetime.strptime(dt_this_str, '%Y%m%d%H')

    lpo_num_this = lpo_id - 10000*(lpo_id//10000)
    fn_this = dt_this.strftime('%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc')
    fn = f'{data_dir_lpo}/{fn_this}'

    with xr.open_dataset(fn) as ds:
        this_pixels_x = ds['pixels_x'][lpo_num_this,:].data
        this_pixels_y = ds['pixels_y'][lpo_num_this,:].data

    # These are stored as int values. Therefore, I cannot use np.isfinite.
    this_pixels_x = this_pixels_x[this_pixels_x > -1]
    this_pixels_y = this_pixels_y[this_pixels_y > -1]

    return (this_pixels_x, this_pixels_y)

def process_an_lpo(lpo_id):
    datetime_to_get_str = str(int(lpo_id))[0:10]
    datetime_to_get = dt.datetime.strptime(datetime_to_get_str, '%Y%m%d%H')
    i_to_get, j_to_get = get_points_from_lpo(lpo_id)
    return np.nanmax(get_ivt_at_a_time_and_point(datetime_to_get, i_to_get, j_to_get))

#generate output file

instance = int(sys.argv[1])
output_file = '/home/disk/orca/csmall3/AR_testing_research/Final_AR_Results/text_files/Total_Global_ARs_'+str(instance)+'.csv'

# Write the header only once (outside the loop)
AR_PAC_df = pd.DataFrame({'AR ID (Number)': [], 'End Season Year': [], 'Max IVT': [], 'AR ID (string)': []})  # Create an empty dataframe
AR_PAC_df.to_csv(output_file, index=False) 

#initialize variables
ar_id = []
ar_fin_ivt = []
ar_year = []

if __name__ == '__main__':

    # Give the year on the command line. E.g., 2023 for June 2023 - June 2024.
    # Also give the number of CPUs.
    # Example usage: python get_max_ivt.py 2023 32
    # to run 2023 on 32 CPUs (o9 - o12)
    # Use 24 CPUs on o1 - o8.
    year1 = int(sys.argv[1])  #2023
    n_cpu = int(sys.argv[2])  #32

    ############################
    lptid_list = get_lptid_list(year1)
    print(len(lptid_list))

    #get the lptIDs as file strings
    year_2 = 1 + year1
    test_ls = sorted(glob.glob('/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/lpt-python-public/ar.testing/data/ar/g0_0h/thresh1/systems/'+str(year1)+'060100_'+str(year_2)+'063023/*.nc'))


    # I have it doing the first 10 for testing. You can have it do all of them.
    # for lptid in lptid_list[0:10]:
    for lptid in lptid_list:
        print(f'Doing {lptid}.')
        this_lpo_list = get_lpo_list(year1, lptid)

        with Pool(n_cpu) as p:
            max_ivt_by_lpo = list(tqdm.tqdm(
                p.imap(process_an_lpo, this_lpo_list),
                total=len(this_lpo_list),
                desc='Get IVT values'
                )
            )

        print(f'Max IVT for the entire LPT system: {np.nanmax(max_ivt_by_lpo)}.')

        # Write the values to a file here.
        # Once processing is done, append the results to the CSV file
        AR_PAC_df = pd.DataFrame({'AR ID (Number)': [str(lptid)], 
                                'End Season Year': [str(instance)], 
                                'Max IVT': [np.nanmax(max_ivt_by_lpo)],
                                'AR ID (string)':[str(test_ls[int(lptid)])]})
        
        # Append the new data without writing the header again
        AR_PAC_df.to_csv(output_file, mode='a', header=False, index=False)
        
        # print(f'finished {idx} of {len(ls_contents)}')

    print('Done.')
