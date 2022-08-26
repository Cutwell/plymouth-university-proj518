import pandas as pd
import os

os.chdir('C:\\Users\\zacha\\Documents\\GitHub\\plymouth-university-proj518\\data')
print(os.getcwd())

df = pd.read_csv('UK east coast 2022 velocity potential .995 sigma.csv')

df_2 = pd.read_csv('UKHO ADMIRALTY bathymetry UK east coast.csv')

# merge the two dataframes
df = pd.merge(df, df_2, on=['Lon', 'Lat'])

# save as UK east coast.csv
df.to_csv('UK east coast velocity potential bathymetry.csv', index=False)