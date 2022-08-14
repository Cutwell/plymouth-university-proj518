import pandas as pd
from inpoly import inpoly2
import numpy as np

# load data/UK wind energy locations raw.csv
df = pd.read_csv('data/UK wind energy locations raw.csv')

# remove all columns except 'LAT', 'LNG', 'EASTING', 'NORTHING', 'TURBNUM'
df = df[['LAT', 'LNG', 'EASTING', 'NORTHING', 'TURBNUM']]

# load search space polygon from data/UK east coast edge points alpha 17.5
polydf = pd.read_csv('data/UK east coast edge points alpha 17.5.csv')

# convert polydf 'Lon', 'Lat' to numpy array
poly = np.array([polydf['Lon'], polydf['Lat']]).T

# get numpy array of 'LNG', 'LAT' from df
points = np.array([df['LNG'], df['LAT']]).T

#get boolean mask for points if in or on polygon perimeter
isin, ison = inpoly2(points, poly)

# mask points if they are in or on polygon perimeter
df = df[isin | ison]

# save as data/UK wind energy locations.csv
df.to_csv('data/UK wind energy locations.csv', index=False)