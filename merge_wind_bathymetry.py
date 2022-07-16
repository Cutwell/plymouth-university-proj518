import pandas as pd

# open admiralty bathymetry UK east coast.csv
df = pd.read_csv('data/admiralty bathymetry UK east coast.csv')

# open UK east coast velocity potential .995 sigma.csv
df_2 = pd.read_csv('data/UK east coast velocity potential .995 sigma.csv')

# merge the two dataframes
df = pd.merge(df, df_2, on=['Lat', 'Lon'])

# save as UK east coast.csv
df.to_csv('data/UK east coast.csv', index=False)