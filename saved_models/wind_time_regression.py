import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# set working directory to project root
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


def scaler():
    file = "data/NCEP daily velocity potential reanalysis.csv"
    df = pd.read_csv(file, sep=",")

    x = df[["Lon", "Lat", "Day"]]
    y = df[["Chi"]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    yscaler = StandardScaler()
    yscaler = yscaler.fit(y)

    # transform y_train and y_test to standardized scale
    y_train = yscaler.transform(y_train)
    y_test = yscaler.transform(y_test)

    scaler_path = os.path.join(
        os.getcwd(),
        "saved_models",
        f"wind_time_scaler_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}",
    )
    joblib.dump(yscaler, scaler_path)
    print("Saved scaler at %s " % scaler_path)

    # normalise X input type
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, y_train, x_test, y_test, yscaler


def build():
    model = Sequential()

    model.add(Dense(32, input_dim=3, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(512, activation="relu"))  # NEW
    model.add(Dense(64, activation="relu"))  # NEW
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.summary()

    model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])

    return model


def train(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=150,
    batch_size=50,
    verbose=1,
    validation_split=0.2,
):

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_split=validation_split,
    )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    # save model and weights
    model_path = os.path.join(
        os.getcwd(),
        "saved_models",
        f"wind_time_regression_model_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.h5",
    )
    model.save(model_path)
    print("Saved trained model at %s " % model_path)

    # score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

    return model


def test(model, yscaler, coords):
    regression = model.predict(coords)
    regression = yscaler.inverse_transform(
        regression
    )  # transform back to original scale

    regression = pd.DataFrame(regression, columns=["Chi"])
    regression = pd.concat([coords, regression], axis=1)

    print(regression)


def main():
    # build model
    x_train, y_train, x_test, y_test, yscaler = scaler()
    model = build()

    # train model
    model = train(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=200,  # test 1-5: 150, test 6: 200
        batch_size=512,  # test 1-5: 50, test 6: 128
        verbose=1,
        validation_split=0.1,
    )

    # print model summary - architecture
    model.summary()

    # test model
    test_coords = pd.DataFrame([[-37.814, 144.96332, 1], [-37.814, 144.96332, 103]])
    test_coords.columns = ["Lon", "Lat", "Day"]
    test(model, yscaler, coords=test_coords)


if __name__ == "__main__":
    main()
