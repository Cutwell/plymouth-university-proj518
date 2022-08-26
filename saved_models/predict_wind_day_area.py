import tensorflow as tf
import os
from os.path import exists
from tensorflow.python.keras.models import load_model
import pandas as pd
import joblib
import loadingbar

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# ignore warning for assigning entire column to a scalar value
pd.set_option('mode.chained_assignment', None)

os.chdir('C:/Users/zacha/Documents/GitHub/plymouth-university-proj518')
print(os.getcwd())

def batch(i, o, days=(0, 228+1), model_name='wind_time_regression_model_25082022164000.h5', scaler_name='wind_time_scaler_25082022164000', saved_models='saved_models'):
    folder = i
    output = o

    save_dir = os.path.join(os.getcwd(), saved_models)
    model_path = os.path.join(save_dir, model_name)

    if exists(model_path):  # load model if exists
        print(f"LOADING MODEL: {model_path}")
        model = load_model(model_path)
    else:
        raise Exception("NO MODEL FOUND")

    model.summary()

    # load wind model input scaler
    save_dir = os.path.join(os.getcwd(), saved_models)
    scaler_path = os.path.join(save_dir, scaler_name)

    scaler = joblib.load(scaler_path)

    # read file names from folder
    file_names = os.listdir(folder)

    days_list = [str(i) for i in range(days[0], days[1])]

    # only predict for datapoints within our search space
    lat_range = (53.486257927, 54.1)
    lon_range = (0.5, 2.5571098)

    # for each file in folder
    for file in file_names:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, sep=' ')    # read file

        filedata = pd.DataFrame(columns=['Lon', 'Lat', *days_list])

        coords = df.drop(columns=['Depth'])     # drop Depth from dataframe

        # filter dataframe to only include data within our search space
        coords = coords[(coords['Lat'] >= lat_range[0]) & (coords['Lat'] <= lat_range[1])]
        coords = coords[(coords['Lon'] >= lon_range[0]) & (coords['Lon'] <= lon_range[1])]
        
        print(f"File: {file_path}, Datapoints: {df.shape[0]}")
        L = loadingbar.PercentageInfoLoadingBar(days[1])

        # fill lat lon datapoints in filedata with df values
        filedata['Lon'] = coords['Lon']
        filedata['Lat'] = coords['Lat']

        X = filedata[['Lon', 'Lat']]

        # ensure X size is non-zero
        if X.shape[0] > 0:
            # iterate through each row in dataframe
            for day in range(days[0], days[1]):
                L.update(1)

                # set day to current day for all rows
                X['Day'] = day

                # predict in batches (per file)
                regression = model.predict(X)
                regression = scaler.inverse_transform(regression)  # transform back to original scale

                regression = pd.DataFrame(regression, columns=[str(day)])  # convert predictions to dataframe with column name = day

                # we can add the predictions to the filedata dataframe without reference to lat/lon, as order is retained
                filedata[str(day)] = regression[str(day)] # add predictions to filedata

        L.done()

        output_path = os.path.join(output, file)
        filedata.to_csv(output_path, index=False)

def concat_files(io, days=(0, 228+1)):
    folder = io
    # combine all files in folder into one dataframe
    days_list = [str(i) for i in range(days[0], days[1])]
    folderdata = pd.DataFrame(columns=['Lon', 'Lat', *days_list])

    file_names = os.listdir(folder)

    print("Concatenating files")
    L = loadingbar.PercentageInfoLoadingBar(len(file_names))

    for file in file_names:
        L.update(1)

        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, sep=',')    # read file

        # skip if empty
        if df.shape[0] > 0:
            folderdata = pd.concat([folderdata, df], ignore_index=True)   # concat to output dataframe

    L.done()

    folderdata.to_csv(f'{folder}.csv', index=False) # write to csv file

if __name__ == "__main__":
    batch(i='data/UKHO ADMIRALTY bathymetry UK east coast', o='data/UK east coast 2022 velocity potential .995 sigma')
    concat_files(io='data/UK east coast 2022 velocity potential .995 sigma')