from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

def batch(folder):
    # read file names from 'data\2020 velocity potential .995 sigma' folder
    file_names = os.listdir(folder)
    print(file_names)

    # create a csv mapping coordinates to wind speed
    with open(f"{folder}.csv", 'w') as f:
        # write headers
        f.write(f"Lat,Lon,Chi\n")

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

def single(path):
    with open(f"{path}.csv", 'w') as f:
        # write headers
        f.write(f"Lat,Lon,Chi\n")

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
    #batch('data/2020 velocity potential .995 sigma')

    single('data/2020 velocity potential .995 sigma/jan2020.nc')