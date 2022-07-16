import numpy as np
import os
import pandas as pd

def batch(folder):
    # read file names from folder
    file_names = os.listdir(folder)
    print(file_names)

    # create empty dataframe
    df = pd.DataFrame(columns=['Lat', 'Lon', 'Depth'])

    for file in file_names:
        file_path = os.path.join(folder, file)
        print(file_path)
        data = pd.read_csv(file_path, sep=' ')

        # append data to dataframe
        df = pd.concat([df, data], ignore_index=True)

    df.to_csv(f'{folder}.csv', index=False)

if __name__ == "__main__":
    batch('data/admiralty bathymetry UK east coast')