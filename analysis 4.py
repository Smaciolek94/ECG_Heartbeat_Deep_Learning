### Version 4:
#Removing the automatic function to allow more tweaking of the models
#fixing convolution models by adding flatten layers

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

#number of times to run each model
ntries = 1
nepochs = 50

#importing, appending, shuffling, splitting into test and train sets
normal = pd.read_csv("C:\\Users\\smaci\\Documents\\GitHub\\ECG_Heartbeat_Deep_Learning\\ptbdb_normal.csv"
                     ,header=None)
abnormal = pd.read_csv("C:\\Users\\smaci\\Documents\\GitHub\\ECG_Heartbeat_Deep_Learning\\ptbdb_abnormal.csv"
                     ,header=None)
total = pd.concat([normal,abnormal])
#total = shuffle(total)
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
base_acc = abnormal.shape[0] / (abnormal.shape[0] + normal.shape[0])

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

##Adding a call back to kill the process when a specified accuracy is hit
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
       if(logs.get('accuracy')>0.99):
           print("\nReached 99% accuracy so cancelling training!")
           self.model.stop_training = True
callbacks = myCallback()

#adding a quick function to test the model 5 times
#when calling it, do not put parentheses around the model function
#EG: tester(A), even though to call A do A()
def tester(model):
    accuracy = []
    Num = []
    for i in range(0,5):
        acc = model()
        if acc >= base_acc:
            num=1
        else:
            num=0
        accuracy.append(acc)
        Num.append(num)
    return(stat.mean(accuracy),stat.stdev(accuracy),sum(Num))

#Model A: 1 hidden layer w/64 nodes, sparsecategorical cross entropy
def A():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(128,activation="relu")(Input)
    Output = layers.Dense(2)(Layer1)
    ModelA = keras.Model(inputs=Input,outputs=Output)
    ModelA.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelA.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predictA = ModelA.predict(xtest)
    PredictA = np.argmax(predictA,axis=1)
    cmA = confusion_matrix(PredictA,ytest)
    accuracyA = stat.mean(PredictA == ytest)
    return(accuracyA)
#accuracies were closer to .6, now closer to .8
#accruacy improves greatly with more epochs
#increasing number of nodes doesn't help much


#Model B: 2 hidden layers w/64,64 nodes, sparsecategorical cross entropy
AccuracyB = []
for i in range(0,ntries):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(128,activation="relu")(Input)
    Layer2 = layers.Dense(64,activation="relu")(Layer1)
    Output = layers.Dense(2)(Layer2)
    ModelB = keras.Model(inputs=Input,outputs=Output)
    ModelB.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelB.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predictB = ModelB.predict(xtest)
    PredictB = np.argmax(predictB,axis=1)
    cmB = confusion_matrix(PredictB,ytest)
    accuracyB = stat.mean(PredictB == ytest)
    AccuracyB.append(accuracyB)
print(AccuracyB)
#accuracies were closer to .6, now closer to .8
#increasing epochs and nodes gets us around 97% accuracy

#Model C: 1 hidden layer w/64 nodes, binary  cross entropy
AccuracyC = []
for i in range(0,ntries):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(64,activation="relu")(Input)
    Output = layers.Dense(2)(Layer1)
    ModelC = keras.Model(inputs=Input,outputs=Output)
    ModelC.compile(
        loss=keras.losses.binary_crossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelC.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy())
    ModelC.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predictC = ModelC.predict(xtest)
    PredictC = np.argmax(predictC,axis=1)
    cmC = confusion_matrix(PredictC,ytest)
    accuracyC = stat.mean(PredictC == ytest)
    AccuracyC.append(accuracyC)
print(AccuracyC)
#aacuracies vary widely

#Model D: 3 hidden layers w/64,64,64 nodes, sparsecategorical cross entropy
AccuracyD = []
for i in range(0,ntries):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187))
    Layer1 = layers.Dense(512,activation="relu")(Input)
    Layer2 = layers.Dense(256,activation="relu")(Layer1)
    Layer3 = layers.Dense(128,activation="relu")(Layer2)
    Output = layers.Dense(2)(Layer3)
    ModelD = keras.Model(inputs=Input,outputs=Output)
    ModelD.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelD.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predictD = ModelD.predict(xtest)
    PredictD = np.argmax(predictD,axis=1)
    cmD = confusion_matrix(PredictD,ytest)
    accuracyD = stat.mean(PredictD == ytest)
    AccuracyD.append(accuracyD)
print(AccuracyD)
#hit 99% accuracy with 64 nodes in 51 epochs
#increasing nodes increase hit 99% accuracy in 46 epochs

#Model E: 1 Convolutional Layer, then 2 hidden dense layers

AccuracyE = []
for i in range(0,ntries):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187,1))
    Layer1 = layers.Conv1D(64,3,activation="relu",input_shape=(None,187))(Input)
    Layer2 = layers.MaxPool1D(2)(Layer1)
    Layer3 = layers.Flatten()(Layer2)
    Layer4 = layers.Dense(64,activation="relu")(Layer3)
    Layer5 = layers.Dense(64,activation="relu")(Layer4)
    Output = layers.Dense(2)(Layer5)
    ModelE = keras.Model(inputs=Input,outputs=Output)
    ModelE.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelE.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predictE = ModelE.predict(xtest)
    PredictE = np.argmax(predictE,axis=1)
    cmE = confusion_matrix(PredictE,ytest)
    accuracyE = stat.mean(PredictE == ytest)
    AccuracyE.append(accuracyE)
print(AccuracyE)
#the accuracy is lower but closer to the accuracy on the test set

#since accuracy is getting better, upgrading the call back
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
       if(logs.get('accuracy')>0.999):
           print("\nReached 99.9% accuracy so cancelling training!")
           self.model.stop_training = True
callbacks = myCallback()
#2 conv layers, 3 dense layers
AccuracyF = []
for i in range(0,ntries):
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    Input = keras.Input(shape=(187,1))
    Layer1 = layers.Conv1D(64,3,activation="relu",input_shape=(None,187))(Input)
    Layer2 = layers.MaxPool1D(2)(Layer1)
    Layer3 = keras.layers.Conv1D(64,3,activation="relu")(Layer2)
    Layer4 = keras.layers.MaxPool1D(2)(Layer3)
    Layer5 = layers.Flatten()(Layer4)
   # Layer6 = layers.Dense(64,activation="relu")(Layer5)
   # Layer7 = layers.Dense(64,activation="relu")(Layer6)
    Output = layers.Dense(2)(Layer5)
    ModelF = keras.Model(inputs=Input,outputs=Output)
    ModelF.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    ModelF.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predictF = ModelF.predict(xtest)
    PredictF = np.argmax(predictF,axis=1)
    cmE = confusion_matrix(PredictF,ytest)
    accuracyF = stat.mean(PredictF == ytest)
    AccuracyF.append(accuracyF)
print(AccuracyF)
#64 filters in both layers gets us around 98% accuracy
#adding more filters doesn't help
#removing 1 dense layer didn't change much
#removing 2 layers brings acrrucay down to near 97

print("Categorical Cross Entropy Loss: 1 Layer, Logit = T")
print(AccuracyA)
print("Categorical Cross Entropy Loss: 2 Layers, Logit = T")
print(AccuracyB)
print("Binary Cross Entropy Loss: 1 Layer")
print(AccuracyC)
print("Categorical Cross Entropy: 3 Layers, Logit = T")
print(AccuracyD)
print("Categorical Cross Entropy Loss: 2 Layers, 1 conv layer")
print(AccuracyE)