import pandas as pd
from keras.models import Sequential
from keras.layers import *
import numpy as np

def trainModel(dataContainer, saveModelName):

    w = 92
    h = 112
    d = 3

    # Define the model
    model = Sequential()
    # https://keras.io/layers/convolutional/#conv2d

    # model.add(Conv2D(filters=20, kernel_size=2, activation='relu',input_shape=(h,w,d)))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=40, kernel_size=3, activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Flatten())
    # model.add(Dense(units=30912, activation='relu',input_shape=(h,w)))
    model.add(Dense(units=30912, activation='relu',input_dim=(h*w)))
    model.add(Dense(units=3036, activation='relu'))
    model.add(Dense(units=759, activation='relu'))
    # model.add(Dense(, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(dataContainer.totClasses, activation='softmax'))
    model.add(Dense(units=dataContainer.totClasses, activation='sigmoid'))

    # Compile the model
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    print(np.shape(dataContainer.train))

    # Train the model
    model.fit(
        y = dataContainer.train_labels,
        x = dataContainer.train,
        epochs=3,
        shuffle=True,
        verbose=2
    )

    score = model.evaluate(dataContainer.train, dataContainer.train_labels)
    print(score)

    model.save(saveModelName)
