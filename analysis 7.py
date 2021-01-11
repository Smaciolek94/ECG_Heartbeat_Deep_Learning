#version 7 - biasing towards positive with output layer cut off

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
ntries = 1
nepochs = 50

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
    plt.plot(xtest.iloc[i,])
    if ytest.iloc[i] == 1.0:
        response = "abnormal"
    else:
        response = "normal"
    title = 'ECG ' + response
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

#Model F was the best model with SMOTE
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

def cutofftester(predict,ytest,cutoff):    
    #run from here to test false positive / negative accuracy
    Predict = []
    falseneg = []
    falsepos = []
    for i in range(0,predict.shape[0]):
        if predict[i,0] > cutoff:
            q = 0
        else:
            q = 1
        Predict.append(q)
    Predict = np.array(Predict,dtype=int)
    ytest = np.array(ytest,dtype=int)
    for i in range(0,Predict.shape[0]):
        p = 0
        if Predict[i] == 1 and ytest[i] == 0:
            p = 1
        falsepos.append(p)
    for i in range(0,Predict.shape[0]):
        if Predict[i] == 0 and ytest[i] == 1:
            r = 1
        else:
            r = 0
        falseneg.append(r)
    #Predict0 = np.argmax(predict,axis=1)
    #cm = confusion_matrix(Predict,ytest)
    Accuracy = []
    for i in range(0,Predict.shape[0]):
        gh = 0
        if Predict[i] == ytest[i]:
            gh = 1
        Accuracy.append(gh)
    accuracy = stat.mean(Accuracy)
    falsepositive = stat.mean(falsepos)
    falsenegative = stat.mean(falseneg)
    output = (accuracy,falsepositive,falsenegative,accuracy+falsepositive+falsenegative)
    #return(output)
    return(accuracy)
#no activation seems to be the most accurate

#cutoffs = [-35,-30,-25,-20,-15,-10,-5,0,5,10]
cutoffs = range(-10,26)
CO = []
for cutoff in cutoffs:
    out = cutofftester(predict,ytest,cutoff)
    CO.append(out)
#CO = pd.DataFrame(cutoffs,CO)
#print(CO)
#CO.to_csv("C:\\Users\\smaci\\Documents\\GitHub\ECG_Heartbeat_Deep_Learning\\cutoff_tester2.csv")


plt.plot(cutoffs,CO)
plt.title("Activation Cutoff Vs Model Accuracy")
plt.xlabel("Activation Cutoff")
plt.ylabel("Accuracy")
plt.show()


    
def falsepositive(predict,ytest,cutoff):    
    #run from here to test false positive / negative accuracy
    Predict = []
    falseneg = []
    falsepos = []
    for i in range(0,predict.shape[0]):
        if predict[i,0] > cutoff:
            q = 0
        else:
            q = 1
        Predict.append(q)
    Predict = np.array(Predict,dtype=int)
    ytest = np.array(ytest,dtype=int)
    for i in range(0,Predict.shape[0]):
        p = 0
        if Predict[i] == 1 and ytest[i] == 0:
            p = 1
        falsepos.append(p)
    for i in range(0,Predict.shape[0]):
        if Predict[i] == 0 and ytest[i] == 1:
            r = 1
        else:
            r = 0
        falseneg.append(r)
    #Predict0 = np.argmax(predict,axis=1)
    #cm = confusion_matrix(Predict,ytest)
    Accuracy = []
    for i in range(0,Predict.shape[0]):
        gh = 0
        if Predict[i] == ytest[i]:
            gh = 1
        Accuracy.append(gh)
    accuracy = stat.mean(Accuracy)
    falsepositive = stat.mean(falsepos)
    falsenegative = stat.mean(falseneg)
    output = (accuracy,falsepositive,falsenegative,accuracy+falsepositive+falsenegative)
    #return(output)
    return(falsepositive)
cutoffs = range(-10,26)
COpositive = []
for cutoff in cutoffs:
    out = falsepositive(predict,ytest,cutoff)
    COpositive.append(out)
    
def falsenegative(predict,ytest,cutoff):    
    #run from here to test false positive / negative accuracy
    Predict = []
    falseneg = []
    falsepos = []
    for i in range(0,predict.shape[0]):
        if predict[i,0] > cutoff:
            q = 0
        else:
            q = 1
        Predict.append(q)
    Predict = np.array(Predict,dtype=int)
    ytest = np.array(ytest,dtype=int)
    for i in range(0,Predict.shape[0]):
        p = 0
        if Predict[i] == 1 and ytest[i] == 0:
            p = 1
        falsepos.append(p)
    for i in range(0,Predict.shape[0]):
        if Predict[i] == 0 and ytest[i] == 1:
            r = 1
        else:
            r = 0
        falseneg.append(r)
    #Predict0 = np.argmax(predict,axis=1)
    #cm = confusion_matrix(Predict,ytest)
    Accuracy = []
    for i in range(0,Predict.shape[0]):
        gh = 0
        if Predict[i] == ytest[i]:
            gh = 1
        Accuracy.append(gh)
    accuracy = stat.mean(Accuracy)
    falsepositive = stat.mean(falsepos)
    falsenegative = stat.mean(falseneg)
    output = (accuracy,falsepositive,falsenegative,accuracy+falsepositive+falsenegative)
    #return(output)
    return(falsenegative)
cutoffs = range(-10,26)
COnegative = []
for cutoff in cutoffs:
    out = falsenegative(predict,ytest,cutoff)
    COnegative.append(out)
    
plt.plot(cutoffs,COpositive,label="false positive",color="blue")
plt.plot(cutoffs,COnegative,label="false negative",color="red")
plt.title("False Positive and Negative Rates vs Cutoff")
plt.xlabel("Cutoff")
plt.ylabel("False Positive/Negative Rate")
plt.legend()
plt.show()

plt.plot(cutoffs,COnegative,label="false negative",color="red")
plt.title("False Negative Rate vs Cutoff")
plt.xlabel("Cutoff")
plt.ylabel("False Negative Rate")
plt.show()

plt.plot(cutoffs,COpositive,label=" false positive",color="blue")
plt.title("False Positive vs Cutoff")
plt.xlabel("Cutoff")
plt.ylabel("False Positive Rate")
plt.show()