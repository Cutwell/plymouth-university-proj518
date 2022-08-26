import tensorflow as tf
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import subprocess
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

os.chdir("C:/Users/zacha/Documents/GitHub/plymouth-university-proj518")
print(os.getcwd())

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def load():
    file = "data/2020 velocity potential .995 sigma.csv"
    df = pd.read_csv(file, sep=",")

    x = df[["Lon", "Lat"]]
    y = df[["Chi"]]

    x_train, y_train, x_test, y_test = train_test_split(
        x, y, test_size=0.1, random_state=0
    )

    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()

    model.add(Dense(32, input_dim=3, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(512, activation="relu"))    #NEW
    model.add(Dense(64, activation="relu"))     #NEW
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])

    model.summary()

    return model


def build():
    # wrap CNN in KerasRegressor to use sklearn interface
    clf = KerasRegressor(build_fn=build, verbose=0)

    # create a ML pipeline for input scaling and CNN
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return pipe


def main():
    x_train, y_train, x_test, y_test = load()

    pipe = build()

    # fit data
    pipe.fit(x_train, y_train)

    # get test performance
    print(pipe.score(x_test, y_test))

    # save pipeline using joblib
    pipeline_path = os.path.join(
        os.getcwd(),
        "saved_models",
        f"wind_time_regression_model_{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
    )

    joblib.dump(pipe, pipeline_path)

    # print model summary - architecture
    pipe.summary()

    test_coords = pd.DataFrame([[-37.814, 144.96332], [-37.814, 144.96332]])
    test_coords.columns = ["Lon", "Lat"]
    print(test_coords)
    test(pipe, coords=test_coords)


def test(model, coords):
    regression = model.predict(coords)
    regression = model.inverse_transform(regression)  # transform back to original scale

    regression = pd.DataFrame(regression, columns=["Chi"])
    regression = pd.concat([coords, regression], axis=1)
    print(regression)


if __name__ == "__main__":
    main()
