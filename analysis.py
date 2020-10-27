### Version 4:
#Removing the automatic function to allow m

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import layers
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import statistics as stat

#importing, appending, shuffling, splitting into test and train sets
normal = pd.read_csv("C:\\Users\\smaci\\Documents\\GitHub\\ECG_Heartbeat_Deep_Learning\\ptbdb_normal.csv"
                     ,header=None)
abnormal = pd.read_csv("C:\\Users\\smaci\\Documents\\GitHub\\ECG_Heartbeat_Deep_Learning\\ptbdb_abnormal.csv"
                     ,header=None)
total = pd.concat([normal,abnormal])
#total = shuffle(total)
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)

#plotting some test ECGs
index = [1,10,100,1000,10000]
for i in index:
    plt.plot(xtrain.iloc[i,])
    if ytrain.iloc[i] == 1.0:
        response = "abnormal"
    else:
        response = "normal"
    title = 'ECG ' + str(i) + " " + response
    plt.title(title)
    plt.show()

##WE WANT TO RUN THE MODEL SEVERAL TIMES TO ENSURE SIMILAR ACCURACIES

#Model A: 1 hidden layer w/64 nodes, sparsecategorical cross entropy
AccuracyA = []
for i in range(0,5):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(64,activation="relu")(Input)
    Output = layers.Dense(2)(Layer1)
    ModelA = keras.Model(inputs=Input,outputs=Output)
    ModelA.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelA.fit(xtrain, ytrain, batch_size=64, epochs=5, validation_split=0.2)
    predictA = ModelA.predict(xtest)
    PredictA = np.argmax(predictA,axis=1)
    cmA = confusion_matrix(PredictA,ytest)
    accuracyA = stat.mean(PredictA == ytest)
    AccuracyA.append(accuracyA)
print(AccuracyA)
#accuracies were closer to .6, now closer to .8

#Model B: 2 hidden layers w/64,64 nodes, sparsecategorical cross entropy
AccuracyB = []
for i in range(0,5):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(64,activation="relu")(Input)
    Layer2 = layers.Dense(64,activation="relu")(Layer1)
    Output = layers.Dense(2)(Layer2)
    ModelB = keras.Model(inputs=Input,outputs=Output)
    ModelB.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelB.fit(xtrain, ytrain, batch_size=64, epochs=5, validation_split=0.2)
    predictB = ModelB.predict(xtest)
    PredictB = np.argmax(predictB,axis=1)
    cmB = confusion_matrix(PredictB,ytest)
    accuracyB = stat.mean(PredictB == ytest)
    AccuracyB.append(accuracyB)
print(AccuracyB)
#accuracies were closer to .6, now closer to .8

#Model C: 1 hidden layer w/64 nodes, binary  cross entropy
AccuracyC = []
for i in range(0,5):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(64,activation="relu")(Input)
    Output = layers.Dense(2)(Layer1)
    ModelC = keras.Model(inputs=Input,outputs=Output)
    #ModelC.compile(
    #    loss=keras.losses.binary_crossentropy(from_logits=True),
    #    optimizer = keras.optimizers.RMSprop(),
    #    metrics=["accuracy"],
    #    )
    ModelC.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy())
    ModelC.fit(xtrain, ytrain, batch_size=64, epochs=5, validation_split=0.2)
    predictC = ModelC.predict(xtest)
    PredictC = np.argmax(predictC,axis=1)
    cmC = confusion_matrix(PredictC,ytest)
    accuracyC = stat.mean(PredictC == ytest)
    AccuracyC.append(accuracyC)
print(AccuracyC)
#aacuracies vary widely

#Model D: 3 hidden layers w/64,64,64 nodes, sparsecategorical cross entropy
AccuracyD = []
for i in range(0,5):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(64,activation="relu")(Input)
    Layer2 = layers.Dense(64,activation="relu")(Layer1)
    Layer3 = layers.Dense(64,activation="relu")(Layer2)
    Output = layers.Dense(2)(Layer3)
    ModelD = keras.Model(inputs=Input,outputs=Output)
    ModelD.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelD.fit(xtrain, ytrain, batch_size=64, epochs=5, validation_split=0.2)
    predictD = ModelD.predict(xtest)
    PredictD = np.argmax(predictD,axis=1)
    cmD = confusion_matrix(PredictD,ytest)
    accuracyD = stat.mean(PredictD == ytest)
    AccuracyD.append(accuracyD)
print(AccuracyD)

#Model E: 1 Convolutional Layer, then 2 hidden dense layers

AccuracyE = []
for i in range(0,5):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Conv1D(3,(3),activation="relu",input(shape=(None,187)))(Input)
    Layer2 = layers.Dense(64,activation="relu")(Layer1)
    Layer3 = layers.Dense(64,activation="relu")(Layer2)
    Output = layers.Dense(2)(Layer3)
    ModelE = keras.Model(inputs=Input,outputs=Output)
    ModelE.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelE.fit(xtrain, ytrain, batch_size=64, epochs=5, validation_split=0.2)
    predictE = ModelE.predict(xtest)
    PredictE = np.argmax(predictE,axis=1)
    cmE = confusion_matrix(PredictE,ytest)
    accuracyE = stat.mean(PredictE == ytest)
    AccuracyE.append(accuracyE)
print(AccuracyE)

print("Categorical Cross Entropy Loss: 1 Layer, Logit = T")
print(AccuracyA)
print("Categorical Cross Entropy Loss: 2 Layers, Logit = T")
print(AccuracyB)
print("Binary Cross Entropy Loss: 1 Layer")
print(AccuracyC)
print("Categorical Cross Entropy: 3 Layers, Logit = T")
print(AccuracyD)