from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_nc_mean(folder):
    # read file names from 'data\2020 velocity potential .995 sigma' folder
    file_names = os.listdir(folder)
    print(file_names)

    # create a csv mapping coordinates to wind speed
    with open(f"{folder}.csv", 'w') as f:
        # write headers
        f.write(f"Lon,Lat,Chi\n")

        for file in file_names:
            file_path = os.path.join(folder, file)
            print(file_path)
            data = Dataset(file_path, mode='r') # read the data 

            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]-180.0
            chi = data.variables['chi'][::]

            chi = np.squeeze(chi)
            longs, lats = np.meshgrid(lon,lat)  #this converts coordinates into 2D array

            print(f"longs shape: {longs.shape}, lats shape: {lats.shape}, chi shape: {chi.shape}")

            # write data
            for i in range(longs.shape[0]):
                for j in range(longs.shape[1]):
                    x, y = longs[i][j], lats[i][j]
                    z = chi[i][j]
                    f.write(f"{x},{y},{z}\n")

def convert_nc_daily(folder):
    # read file names from folder and convert daily measurements into single csv file
    file_names = os.listdir(folder)
    print(file_names)

    # create a csv mapping coordinates to wind speed
    with open(f"{folder}.csv", 'w') as f:
        # write headers
        f.write(f"Lon,Lat,Day,Chi\n")

        for file in file_names:
            file_path = os.path.join(folder, file)
            print(file_path)
            data = Dataset(file_path, mode='r') # read the data 

            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]-180.0
            chi = data.variables['chi'][::]
            level = data.variables['level'][:]

            chi = np.squeeze(chi)
            longs, lats = np.meshgrid(lon,lat)  #this converts coordinates into 2D array

            print(f"longs shape: {longs.shape}, lats shape: {lats.shape}, chi shape: {chi.shape}")

            # write data
            level = 0   # default level to .995 sigma
            for i in range(0, chi.shape[2]):   # lon
                for j in range(0, chi.shape[3]):   # lat
                    for k in range(0, chi.shape[0] - 1):   # day
                        x, y = longs[i][j], lats[i][j]
                        z = chi[k][level][i][j]
                        f.write(f"{x},{y},{k},{z}\n")

def single(path):
    with open(f"{path}.csv", 'w') as f:
        # write headers
        f.write(f"Lon,Lat,Chi\n")

        data = Dataset(path, mode='r') # read the data 

        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]-180.0
        chi = data.variables['chi'][::]

        chi = np.squeeze(chi)
        longs, lats = np.meshgrid(lon,lat)  #this converts coordinates into 2D array

        print(f"longs shape: {longs.shape}, lats shape: {lats.shape}, chi shape: {chi.shape}")

        # write data
        for i in range(longs.shape[0]):
            for j in range(longs.shape[1]):
                x, y = longs[i][j], lats[i][j]
                z = chi[i][j]
                f.write(f"{x},{y},{z}\n")

if __name__ == "__main__":
    print(os.getcwd())
    #convert_nc_mean('data/2020 velocity potential .995 sigma')

    convert_nc_daily('data/NCEP daily velocity potential reanalysis')