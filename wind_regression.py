import tensorflow as tf
import os
import matplotlib.pyplot as plt
from os.path import exists
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Dropout
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'wind_regression_model.h5'
scaler_name = 'wind_regression_scaler'

if not os.path.isdir(save_dir):
    raise Exception("Model directory doesn't exist!")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def load(scaler_path=None):
    file = 'data/2020 velocity potential .995 sigma.csv'
    df = pd.read_csv(file, sep=',')

    x = df[['Lat','Lon']]
    y = df[['Chi']]

    #scaler_x = MinMaxScaler()
    #scaler_y = MinMaxScaler()
    #
    #print(scaler_x.fit(x))
    #print(scaler_y.fit(y))
    #
    #xscale=scaler_x.transform(x)
    #yscale=scaler_y.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # open standard scaler if exists
    if scaler_path is not None and exists(scaler_path):
        # load wind model input scaler
        yscaler = joblib.load(scaler_path)
    else:   # else create new scaler
        print("NO SCALER FOUND, BUILDING")
        yscaler = StandardScaler()
        yscaler = yscaler.fit(y)
    
    # transform y_train and y_test to standardized scale
    y_train = yscaler.transform(y_train)
    y_test = yscaler.transform(y_test)
    
    # save if path given
    if scaler_path is not None:
        joblib.dump(yscaler, scaler_path)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # normalise X input type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test, yscaler

def build():
    model = Sequential()

    # test 1
    #model.add(Dense(12, input_dim=2, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(1, activation='linear'))

    # test 2
    #model.add(Dense(32, input_dim=2, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(1, activation='linear'))

    # test 3 (4?)
    model.add(Dense(32, input_dim=2, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    #model.add(Dense(32, input_dim=2, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    ##model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    ##model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    ##model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(1, activation='linear'))
    
    model.summary()

    # initiate RMSprop optimizer
    #opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    # train using RMSprop and cross-entropy loss
    #model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

    return model


def train(model, x_train, y_train, x_test, y_test, epochs=150, batch_size=50, verbose=1, validation_split=0.2, save_dir=None, model_name=None):

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # save model and weights
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    # score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    return model


def test(model, yscaler, coords):
    regression = model.predict(coords)
    regression = yscaler.inverse_transform(regression)  # transform back to original scale

    regression = pd.DataFrame(regression, columns=['Chi'])
    regression = pd.concat([coords, regression], axis=1)
    print(regression)


if __name__ == "__main__":
    
    scaler_path = os.path.join(save_dir, scaler_name)
    x_train, y_train, x_test, y_test, yscaler = load(scaler_path)

    model_path = os.path.join(save_dir, model_name)
    
    if exists(model_path):  # load model if exists
        print(f"LOADING MODEL: {model_path}")

        model = load_model(model_path)

    else:   # else generate new model
        print("NO MODEL FOUND, BUILDING")

        model = build()

        # train for large epoch size
        model = train(
            model, 
            x_train, 
            y_train, 
            x_test, 
            y_test, 
            epochs=200,         # test 1-5: 150, test 6: 200
            batch_size=128,     # test 1-5: 50, test 6: 128
            verbose=1, 
            validation_split=0.2,
            save_dir=save_dir,
            model_name=model_name
        )

    # print model summary - architecture
    model.summary()

    #test_coords = pd.DataFrame([[-37.814, 144.96332]])
    test_coords = pd.DataFrame([[-37.814, 144.96332], [-37.814, 144.96332]])
    test_coords.columns = ['Lat','Lon']
    print(test_coords)

    test(
        model,
        yscaler,
        coords=test_coords
    )
