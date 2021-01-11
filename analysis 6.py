### Version 7:
#adding SMOTE so that there's a 50-50 class balance

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
import random
from imblearn.over_sampling import SMOTE

#number of times to run each model
nepochs = 50

#importing, appending, shuffling, splitting into test and train sets
normal = pd.read_csv("C:\\Users\\smaci\\Documents\\GitHub\\ECG_Heartbeat_Deep_Learning\\ptbdb_normal.csv"
                     ,header=None)
abnormal = pd.read_csv("C:\\Users\\smaci\\Documents\\GitHub\\ECG_Heartbeat_Deep_Learning\\ptbdb_abnormal.csv"
                     ,header=None)
total = pd.concat([normal,abnormal])

#total = shuffle(total)
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],
                total.iloc[:,187],test_size=0.20)
sm = SMOTE()
xtrain_res, ytrain_res = sm.fit_sample(xtrain, ytrain.ravel())
base_acc = sum(ytrain_res) / ytrain_res.shape[0]

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

##Adding a call back to kill the process when a specified accuracy is hit
##stopacc is the accuracy to stop fitting the model at
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
       if(logs.get('accuracy')>.999):
           print("\nReached 99.9% accuracy so cancelling training!")
           self.model.stop_training = True
callbacks = myCallback() #this is an instance of this class

#adding a quick function to test the model 5 times
#when calling it, do not put parentheses around the model function
#EG: tester(A), even though to call A do A()
def tester(model):
    accuracy = []
    Num = []
    for i in range(0,5):
        print("WE ARE ON MODEL")
        print(model)
        print("WE ARE ON ITERATION")
        print(i+1)
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
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(2)
        ])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)

#tester(A)
#accuracies were closer to .6, now closer to .8
#accruacy improves greatly with more epochs
#increasing number of nodes doesn't help much


#Model B: 2 hidden layers w/64,64 nodes, sparsecategorical cross entropy
def B():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(2)
        ])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)

#tester(B)
#accuracies were closer to .6, now closer to .8
#increasing epochs and nodes gets us around 97% accuracy

#Model D: 3 hidden layers w/64,64,64 nodes, sparsecategorical cross entropy
def C():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(2)
        ])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)

#tester(C)
#hit 99% accuracy with 64 nodes in 51 epochs
#increasing nodes increase hit 99% accuracy in 46 epochs

#Model E: 1 Convolutional Layer, then 2 hidden dense layers
    

#the accuracy is lower but closer to the accuracy on the test set


#since this D didn't work, trying something else:
def D():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    Input = keras.Input(shape=(187,1))
    x = layers.Conv1D(64,3,activation="relu",input_shape=(1,187))(Input)
    x = layers.MaxPool1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation="relu")(x)
    x = layers.Dense(64,activation="relu")(x)
    Output = layers.Dense(2)(x)
    model = keras.Model(inputs=Input,outputs=Output)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)
        
 #since this D didn't work, trying something else:
def E():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    Input = keras.Input(shape=(187,1))
    x = layers.Conv1D(64,3,activation="relu",input_shape=(1,187))(Input)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation="relu")(x)
    x = layers.Dense(64,activation="relu")(x)
    Output = layers.Dense(2)(x)
    model = keras.Model(inputs=Input,outputs=Output)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)
#64 filters in both layers gets us around 98% accuracy
#adding more filters doesn't help
#removing 1 dense layer didn't change much
#removing 2 layers brings acrrucay down to near 97


def F():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    Input = keras.Input(shape=(187,1))
    x = layers.Conv1D(64,3,activation="relu",input_shape=(1,187))(Input)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation="relu")(x)
    x = layers.Dense(64,activation="relu")(x)
    Output = layers.Dense(2)(x)
    model = keras.Model(inputs=Input,outputs=Output)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)

#tester(F)

def G():
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(total.iloc[:,0:187],total.iloc[:,187],test_size=0.20)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain.ravel())
    Input = keras.Input(shape=(187,1))
    x = layers.Conv1D(64,3,activation="relu",input_shape=(1,187))(Input)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64,3,activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation="relu")(x)
    x = layers.Dense(64,activation="relu")(x)
    x = layers.Dense(64,activation="relu")(x)
    Output = layers.Dense(2)(x)
    model = keras.Model(inputs=Input,outputs=Output)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics=["accuracy"],
        )
    model.fit(xtrain, ytrain, batch_size=64, epochs=nepochs, validation_split=0.2,
               callbacks=[callbacks])
    predict = model.predict(xtest)
    Predict = np.argmax(predict,axis=1)
    cm = confusion_matrix(Predict,ytest)
    accuracy = stat.mean(Predict == ytest)
    print(base_acc)
    print(sum(Predict)/Predict.shape[0])
    return(accuracy)

#tester(G)
#this runs each model and compiles average and final accuracy - 1250 total epochs
models = [A,B,C,D,E,F,G]
Lables = ["A","B","C","D","E","F","G"]
final_accuracy_50 = []
for model in models:
    acc = tester(model)
    final_accuracy_50.append(acc)
final_accuracy_50 = pd.DataFrame(Lables,final_accuracy_50)

#recompile functions after this to reset the n epochs

nepochs = 25
models = [A,B,C,D,E,F,G]
Lables = ["A","B","C","D","E","F","G"]
final_accuracy_25 = []
for model in models:
    acc = tester(model)
    final_accuracy_25.append(acc)
final_accuracy_25 = pd.DataFrame(Lables,final_accuracy_25)
print(final_accuracy_25)

print(final_accuracy_50)

final_accuracy_25.to_csv("C:\\Users\\smaci\\Documents\\GitHub\ECG_Heartbeat_Deep_Learning\\accuracy_25_SMOTE.csv")
final_accuracy_50.to_csv("C:\\Users\\smaci\\Documents\\GitHub\ECG_Heartbeat_Deep_Learning\\accuracy_50_SMOTE.csv")

###############################################################################
