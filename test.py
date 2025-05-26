import pandas as pd
pd_05_init = pd.read_csv("E:/Weight_predictor/option-chain-ED-NIFTY-05-Jun-2025.csv", na_values=['-'])
pd_19_init=pd.read_csv("E:/Weight_predictor/option-chain-ED-NIFTY-19-Jun-2025.csv",  na_values=['-'])
# Clean the data by dropping rows where all elements are NaN
pd_05_init = pd_05_init.drop(['unnamed1', 'unnamed2'], axis=1)
pd_19_init = pd_19_init.drop(['unnamed1', 'unnamed2'], axis=1)
pd_05_init.dropna(inplace=True)
pd_19_init.dropna(inplace=True)
pd_05 = pd_05_init
pd_19 = pd_19_init
price_list_1 = pd_05['LTP_call'].to_numpy()
price_list_2 = pd_19['LTP_call'].to_numpy()
print(price_list_1)
print(price_list_2)