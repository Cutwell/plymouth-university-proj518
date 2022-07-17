# generate polygon coordinate points from component bathymetric file lat/lon coordinates
import pandas as pd
import os
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import geopandas as gpd
from scipy.spatial.distance import cdist

def main(folder):
    # get files in folder
    file_names = os.listdir(folder)

    # create empty dataframe to store data
    data = pd.DataFrame(columns=['Lat', 'Lon'])

    # iterate and open each file
    for file in file_names:
        # open file
        file_path = os.path.join(folder, file)
        print(file_path)
        df = pd.read_csv(file_path, sep=' ')

        # drop depth column
        df = df.drop(columns=['Depth'])

        # get corners
        corners = np.array([df[df['Lat'] == df['Lat'].max()].values[0], df[df['Lat'] == df['Lat'].min()].values[0], df[df['Lon'] == df['Lon'].max()].values[0], df[df['Lon'] == df['Lon'].min()].values[0]])

        # create dataframe of coordinates
        coords = pd.DataFrame(corners, columns=['Lat', 'Lon'])

        # add max and min lat and long to dataframe as new rows
        data = pd.concat([data, coords])
    
    data = data.reset_index(drop=True)

    return data

def oneshot(file):
    # open bathymetry coordinates file
    df = pd.read_csv(file)
        
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Lat'], df['Lon']))

    polygon_coords = []
    for point in gdf.geometry.values:
        polygon_coords.append([point.x, point.y])
    
    print("1")

    polygon_coords = np.array(polygon_coords)
    
    polygon_coords = get_ruler(polygon_coords)

    print("2")

    p = Polygon(polygon_coords)
    return p

def dominates(a, b):
    return (np.asarray(a) <= b).all()

def get_ruler(a):
    X = cdist(a, a, metric=dominates).astype(np.bool)
    rulers = np.where(X.all(axis=1))[0]
    if rulers.size > 0:
        return [a[i] for i in rulers]
    else: # no one dominates every other
        return a

if __name__ == "__main__":
    #data = main('data/admiralty bathymetry UK east coast')
    #data=data.to_numpy()
    #poly = Polygon(data)

    poly = oneshot('data/admiralty bathymetry UK east coast.csv')

    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = pd.read_csv('data/UK east coast.csv')
    map = ax.scatter(data['Lat'], data['Lon'], c=data['Depth'], cmap='viridis')

    ax.add_patch(PolygonPatch(poly, fc='#009900', alpha=1))

    ax.set_xlabel('Lat')
    ax.set_ylabel('Lon')
    plt.show()
    