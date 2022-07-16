import pandas as pd
from matplotlib import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

bathymetric_data = pd.read_csv('data/admiralty bathymetry UK east coast.csv')

x = [53.4, 1.6, 52.6, -1.0, -52.6, -1.6, -53.4, 0.5]

coords = pd.DataFrame(columns=['Lat', 'Lon'])
coords['Lat'] = x[0::2] # set every other element to lat (from index 0)
coords['Lon'] = x[1::2] # set every other element to long (from index 1)

print(coords)

# sort coords so they form a bounding box
coords = coords.sort_values(by=['Lat', 'Lon'])
coords = coords.reset_index(drop=True)

print(coords)

# difference between each coordinate and the next
first = coords['Lat'][0]*coords['Lon'][1] + coords['Lat'][1]*coords['Lon'][2] + coords['Lat'][2]*coords['Lon'][3] + coords['Lat'][3]*coords['Lon'][0]
second = coords['Lat'][1]*coords['Lon'][0] + coords['Lat'][2]*coords['Lon'][1] + coords['Lat'][3]*coords['Lon'][2] + coords['Lat'][0]*coords['Lon'][3]
area = 0.5 * (first - second)

print(area)

polygon = path.Path([[coords['Lat'][0], coords['Lon'][0]], [coords['Lat'][1], coords['Lon'][1]], [coords['Lat'][2], coords['Lon'][2]], [coords['Lat'][3], coords['Lon'][3]]])

# get bathymetric data within polygon
bathymetric_data_polygon = bathymetric_data[polygon.contains_points(bathymetric_data[['Lat', 'Lon']].values)]

# get average depth from bathymetric data polygon
depth = bathymetric_data_polygon['Depth'].mean()

# if depth is 0, because no data at that location, set depth to -inf
if depth == 0:
    depth = -np.inf

print(bathymetric_data_polygon)

p1 = Polygon(np.array([coords.loc[0], coords.loc[1], coords.loc[2]]), closed=False)
p2 = Polygon(np.array([coords.loc[3], coords.loc[2], coords.loc[0]]), closed=False)

ax = plt.gca()

ax.add_patch(p1)
ax.add_patch(p2)

ax.set_xlim(-60, 60)
ax.set_ylim(-3, 3)

plt.show()