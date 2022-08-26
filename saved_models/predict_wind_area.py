import tensorflow as tf
import os
from os.path import exists
from tensorflow.python.keras.models import load_model
import pandas as pd
import joblib

def batch(folder):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'wind_regression_model.h5'
    model_path = os.path.join(save_dir, model_name)

    if exists(model_path):  # load model if exists
        print(f"LOADING MODEL: {model_path}")
        wind_model = load_model(model_path)
    else:
        raise Exception("NO MODEL FOUND")

    wind_model.summary()

    # load wind model input scaler
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    scaler_name = 'wind_regression_scaler'
    scaler_path = os.path.join(save_dir, scaler_name)

    wind_scaler = joblib.load(scaler_path)

    # read file names from 'data\admiralty bathymetry UK east coast' folder
    file_names = os.listdir(folder)
    print(file_names)

    wind_data = pd.DataFrame(columns=['Lon', 'Lat', 'Chi'])

    # for each file in folder
    for file in file_names:
        file_path = os.path.join(folder, file)
        print(file_path)
        df = pd.read_csv(file_path, sep=' ')    # read file

        coords = df.drop(columns=['Depth'])     # drop Depth from dataframe

        # predict in batches (per file)
        regression = wind_model.predict(coords)
        regression = wind_scaler.inverse_transform(regression)  # transform back to original scale

        regression = pd.DataFrame(regression, columns=['Chi'])  # convert predictions to dataframe
        regression = pd.concat([coords, regression], axis=1)     # concat coords and regression

        # save as individual files
        output_path = os.path.join('data/UK east coast velocity potential .995 sigma', file)
        regression.to_csv(output_path, index=False)

        wind_data = pd.concat([wind_data, regression], ignore_index=True)   # concat to output dataframe

    wind_data.to_csv('data/wind_data.csv', index=False) # write to csv file

if __name__ == "__main__":
    batch('data/admiralty bathymetry UK east coast')