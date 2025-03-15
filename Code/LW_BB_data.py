import numpy as np
import pandas as pd
import os

# Yield curves from Liu and Wu
path = os.getcwd()  
data_path = os.path.join(os.path.dirname(path), 'Data') 
lw_monthly_yld_curves = pd.read_csv(data_path + "/YCData/LW_monthly.csv", delimiter = ',')
lw_monthly_yld_curves['Date'] = pd.to_datetime(lw_monthly_yld_curves['Date'], format='%Y%m', dayfirst=True) + pd.offsets.MonthEnd(1)

lw_yld_curves_all_months = lw_monthly_yld_curves.copy()
months = list(np.arange(1,361, 1))
lw_yld_curves_all_months.columns = ['Date'] + months

years = list(np.arange(12,372, 12))
lw_yld_curves_all_years = lw_yld_curves_all_months[['Date'] + years]
lw_yld_curves_all_years.columns = ['Date'] + list(np.arange(1,31,1))

# In-sample og out-of-sample data 
lw_yld_curves_all_years_ins = lw_yld_curves_all_years[(lw_yld_curves_all_years['Date'] >= '2010-01-01') & (lw_yld_curves_all_years['Date'] <= '2020-12-31')] 
lw_yld_curves_all_years_ins = lw_yld_curves_all_years_ins.reset_index(drop=True) 
lw_yld_curves_all_years_ins.iloc[:, 1:] = lw_yld_curves_all_years_ins.iloc[:,1:] / 100

lw_yld_curves_all_years_outs = lw_yld_curves_all_years[((lw_yld_curves_all_years['Date'] >= '2021-01-01') & (lw_yld_curves_all_years['Date'] <= '2023-06-30'))] 
lw_yld_curves_all_years_outs = lw_yld_curves_all_years_outs.reset_index(drop=True)
lw_yld_curves_all_years_outs.iloc[:, 1:] = lw_yld_curves_all_years_outs.iloc[:,1:] / 100

# lw_yld_curves_all_years_ins # 132 rows
# lw_yld_curves_all_years_outs # 30 rows

# In- and out-of-sample 'calculated' LW swap rates
numeric_columns_ins = lw_yld_curves_all_years_ins.select_dtypes(include=[np.number])
numeric_columns_outs = lw_yld_curves_all_years_outs.select_dtypes(include=[np.number])

lw_implied_prices_ins = numeric_columns_ins.apply(lambda row: np.exp(-row * numeric_columns_ins.columns.astype(float)), axis=1)
lw_implied_prices_outs = numeric_columns_outs.apply(lambda row: np.exp(-row * numeric_columns_outs.columns.astype(float)), axis=1)
    
lw_implied_prices_cumsum_ins = np.cumsum(lw_implied_prices_ins, axis=1) 
lw_implied_prices_cumsum_outs = np.cumsum(lw_implied_prices_outs, axis=1) 

lw_swap_rates_ins = (1 - lw_implied_prices_ins) / lw_implied_prices_cumsum_ins
lw_swap_rates_outs = (1 - lw_implied_prices_outs) / lw_implied_prices_cumsum_outs

lw_swap_rates_ins = pd.DataFrame(lw_swap_rates_ins)
lw_swap_rates_ins['Date'] = lw_yld_curves_all_years_ins['Date']
lw_swap_rates_ins = lw_swap_rates_ins[['Date'] + [col for col in lw_swap_rates_ins.columns if col != 'Date']]
lw_swap_rates_ins.columns = lw_swap_rates_ins.columns.astype(str)
lw_swap_rates_ins = lw_swap_rates_ins[['Date', '1', '2', '3', '5', '10', '15', '20', '30']]

lw_swap_rates_outs = pd.DataFrame(lw_swap_rates_outs)
lw_swap_rates_outs['Date'] = lw_yld_curves_all_years_outs['Date']
lw_swap_rates_outs = lw_swap_rates_outs[['Date'] + [col for col in lw_swap_rates_outs.columns if col != 'Date']]
lw_swap_rates_outs.columns = lw_swap_rates_outs.columns.astype(str)
lw_swap_rates_outs = lw_swap_rates_outs[['Date', '1', '2', '3', '5', '10', '15', '20', '30']]

# Bloomberg data in-sample and out-of-sample
path = os.getcwd()
data_path = os.path.join(os.path.dirname(path), 'Data') 

BloombergData = pd.read_csv(data_path + "/BloombergData_Swap_Features.csv")

# extracting US swap rates
BloombergData_US = BloombergData[(BloombergData['Currency'] == 'US')].copy()
BloombergData_US['Date'] = pd.to_datetime(BloombergData_US['Date'], dayfirst=False)

# Sætter dato til end-of-month
BloombergData_US['Date'] = BloombergData_US['Date'] + pd.offsets.MonthEnd(0)

# scaler data
BloombergData_US.iloc[:,2:] = BloombergData_US.iloc[:,2:] / 100
BloombergData_US = BloombergData_US.reset_index(drop=True)

# In-sample period: 2010-01-01 - 2020-12-31
# Out-of-sample period: 2021-01-01 - 2023-06-30

in_sample_start = '2010-01-01'
in_sample_end = '2020-12-31'

BloombergData_US_insample = BloombergData_US[((BloombergData_US['Date'] >= in_sample_start) & (BloombergData_US['Date'] <= in_sample_end))] # 132 rows

BloombergData_US_outs = BloombergData_US[(BloombergData_US['Date'] > in_sample_end)] # 24 rows

TestData_post = pd.read_csv(data_path + "/TestData_Swap_Features_post.csv")
TestData_post['Date'] = pd.to_datetime(TestData_post['Date'], dayfirst=False)
TestData_US_post = TestData_post[(TestData_post['Currency'] == 'us')].copy()
TestData_US_post

TestData_US_post['Date'] = TestData_US_post['Date'] + pd.offsets.MonthEnd(0)

TestData_US_post.iloc[:,2:] = TestData_US_post.iloc[:,2:] / 100

TestData_US_post = TestData_US_post.reset_index(drop=True)

BloombergData_US_outs = pd.concat([BloombergData_US_outs, TestData_US_post], ignore_index=True)
BloombergData_US_outs['Currency'] = 'US' 
# BloombergData_US_outs # 30 rows

