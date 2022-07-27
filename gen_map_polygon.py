import pandas as pd
import os
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import geopandas as gpd
import logging
import matplotlib.pyplot as plt



from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math
from descartes import PolygonPatch
import fiona
import shapely.geometry as geometry
import pylab as pl

os.chdir('C:/Users/zacha/Documents/GitHub/plymouth-university-proj518/data')
print(os.getcwd())

def polygonbuffer(file):
    # open bathymetry coordinates file
    path = os.path.join(os.getcwd(), file)
    df = pd.read_csv(path)
        
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Lat'], df['Lon']))
    logging.info('GeoDataFrame created')

    polygon_coords = []
    for point in gdf.geometry.values:
        polygon_coords.append([point.x, point.y])

    logging.info('Polygon coordinates transposed')

    poly = Polygon(polygon_coords)
    logging.info('Polygon created')
    
    poly = poly.buffer(0.001)
    logging.info('Polygon buffered')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    path = os.path.join(os.getcwd(), 'UK east coast.csv')
    data = pd.read_csv(path)
    map = ax.scatter(data['Lat'], data['Lon'], c=data['Depth'], cmap='viridis')

    ax.add_patch(PolygonPatch(poly, fc='#009900', alpha=1))

    ax.set_xlabel('Lat')
    ax.set_ylabel('Lon')
    plt.show()

def alpha_shape(coords, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """

    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig

def convexhull(file):
    path = os.path.join(os.getcwd(), file)
    df = pd.read_csv(path)
        
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Lat'], df['Lon']))
    logging.info('GeoDataFrame created')

    polygon_coords = []
    for point in gdf.geometry.values:
        polygon_coords.append([point.x, point.y])

    polygon_coords = np.array(polygon_coords)

    alpha = 17.5
    concave_hull, edge_points = alpha_shape(polygon_coords, alpha=alpha)

    df = pd.DataFrame(list(concave_hull.exterior.coords), columns=['Lat', 'Lon'])
    
    print(df)

    _ = plot_polygon(concave_hull)
    plt.show()

    if input("Save? (y/n) ").lower() == 'y':
        # save edge points to csv
        path = os.path.join(os.getcwd(), f'UK east coast edge points alpha {alpha}.csv')
        df.to_csv(path, index=False)
        logging.info('Edge points saved')

    return concave_hull, edge_points

if __name__ == "__main__":
    #oneshot('admiralty bathymetry UK east coast.csv')
    convexhull('admiralty bathymetry UK east coast.csv')