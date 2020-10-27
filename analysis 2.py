#Version 2: Creating an evaluator function to make it easier to test each model

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
base_acc = abnormal.shape[0] / (abnormal.shape[0] + normal.shape[0])
print(base_acc)

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

#creating a function to evalutate the model to prevent having to loop everytime
def Evaluator(Model):
    Num = []
    Accuracy = []
    for i in range(0,5):
        xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
        Model.fit(xtrain, ytrain, batch_size=64, epochs=5, validation_split=0.2)
        predict = Model.predict(xtest)
        Predict = np.argmax(predict,axis=1)
        cm = confusion_matrix(Predict,ytest)
        accuracy = stat.mean(Predict == ytest)
        if accuracy >= base_acc:
            num=1
        else:
            num=0
        Accuracy.append(accuracy)
        Num.append(num)
    print("run mean: ",stat.mean(Accuracy))
    print("run std: ",stat.stdev(Accuracy))
    print("run better than base: ",sum(Num))
##WE WANT TO RUN THE MODEL SEVERAL TIMES TO ENSURE SIMILAR ACCURACIES

#Model A: 1 hidden layer w/64 nodes, sparsecategorical cross entropy
    
Input = keras.Input(shape=(187))
Layer1 = layers.Dense(64,activation="relu")(Input)
Output = layers.Dense(2)(Layer1)
ModelA = keras.Model(inputs=Input,outputs=Output)
ModelA.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.RMSprop(),
    metrics=["accuracy"],
    )
Evaluator(ModelA)
#accuracies were closer to .6, now closer to .8

#Model B: 2 hidden layers w/64,64 nodes, sparsecategorical cross entropy
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
Evaluator(ModelB)
#accuracies were closer to .6, now closer to .8

#Model C: 1 hidden layer w/64 nodes, binary  cross entropy
Input = keras.Input(shape=(187))
Layer1 = layers.Dense(64,activation="relu")(Input)
Output = layers.Dense(2)(Layer1)
ModelC = keras.Model(inputs=Input,outputs=Output)
ModelC.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy())
Evaluator(ModelC)
#aacuracies vary widely

#Model D: 3 hidden layers w/64,64,64 nodes, sparsecategorical cross entropy
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
Evaluator(ModelD)

#Model E: 1 Convolutional Layer, then 2 hidden dense layers, sparse categorical cross entropy
#the predicted y vector has values other than 0/1
Input = keras.Input(shape=(187,1))
Layer1 = layers.Conv1D(1,3,activation="relu",input_shape=(None,187,1),padding='same')(Input)
Layer2 = layers.MaxPooling1D(1)(Layer1)
Output = layers.Dense(1)(Layer2)
ModelE = keras.Model(inputs=Input,outputs=Output)
ModelE.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)
Evaluator(ModelE)

ModelE.summary()

Input = keras.Input(shape=(187,1))
Layer1 = layers.Conv1D(3,3,activation="relu",input_shape=(None,187,1))(Input)
Layer2 = layers.Dense(1,activation="relu")(Layer1) #dimensionality error unless all dense layers after are 1
Layer3 = layers.Dense(64,activation="relu")(Layer2)
Output = layers.Dense(2)(Layer3)
ModelF = keras.Model(inputs=Input,outputs=Output)
ModelF.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)
Evaluator(ModelF)