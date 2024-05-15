import tensorflow as tf

import Neural_Network_tester as nn
import heavy_ball as hb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#neural network crated with keras
def neural_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(10, batch_size=input_shape), 
        tf.keras.layers.Dense(4,  activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(0.01), bias_regularizer=None, kernel_initializer='glorot_normal'),
        tf.keras.layers.Dense(12, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(0.01), bias_regularizer=None, kernel_initializer='glorot_normal') 
    ])
    return model


if __name__ == "__main__":
    tf.function(reduce_retracing=True)
    rho = 0.001
    # open dataset
    df = pd.read_table('avila-tr.txt', header=None, sep=',')
    # str type for class column
    df[10] = df[10].astype(str)

    # split data into train and test
    X = df.drop(10, axis=1).values
    y = df[10].values

    # one hot encoding of y
    enc = OneHotEncoder(handle_unknown='ignore')
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    #y = np.argmax(y, axis = 1)

    # min max scaling X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.array(X)

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = len(X_train) 

    i = 1
    diff = 0
    while i < 5:        
        #keras model
        model = neural_network(input_shape)
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=False) 

        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1000, batch_size=len(X_train), verbose=False)
        # Evaluate the model
        k_loss, accuracy = model.evaluate(X_train, y_train, batch_size=len(X_train), verbose=False)

        # Define the neural network architecture with our class
        input_size = X_train.shape[1]
        hidden_size = 4
        output_size = 12
        L1= 1e-2

        model = nn.NeuralNetwork(input_size, hidden_size, output_size, L1)

        f,b = hb.Heavy_Ball_Approach(model, X_train, y_train, 0.01, 0.9, 1000)

        diff += np.abs(k_loss - f[-1])

        i +=1
        
    #average difference
    print(diff/5)