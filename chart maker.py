import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

accs = pd.read_csv("C:/Users/smaci/Documents/GitHub/ECG_Heartbeat_Deep_Learning/accuracies.csv")
accs25 = pd.read_csv("C:/Users/smaci/Documents/GitHub/ECG_Heartbeat_Deep_Learning/accuracies_25.csv")
accsREDUCED = pd.read_csv("C:/Users/smaci/Documents/GitHub/ECG_Heartbeat_Deep_Learning/accuracies_reduced.csv")
accsSMOTE = pd.read_csv("C:/Users/smaci/Documents/GitHub/ECG_Heartbeat_Deep_Learning/accuracies_SMOTE.csv")



#Just the raw models by epochs
plt.plot(accs.iloc[0:7,0],accs.iloc[0:7,1],marker="o",label = "50 Epochs")
#plt.plot(accs.iloc[:,0],accs.iloc[:,1] + 2*accs.iloc[:,2],linewidth=1,color="blue")
#plt.plot(accs.iloc[:,0],accs.iloc[:,1] - 2*accs.iloc[:,2],linewidth=1,color="blue")
plt.plot(accs25.iloc[0:7,0],accs25.iloc[0:7,1],marker="o",color = "red",label="25 Epochs")
#plt.plot(accs25.iloc[:,0],accs25.iloc[:,1] + 2*accs25.iloc[:,2],linewidth=1,color="red")
#plt.plot(accs25.iloc[:,0],accs25.iloc[:,1] - 2*accs25.iloc[:,2],linewidth=1,color="red")
plt.legend()
plt.title("Raw Model Accuracy: 25 vs 50 epochs")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

#plotting just 50 epocs with error
plt.plot(accs.iloc[0:7,0],accs.iloc[0:7,1],marker="o",label = "50 Epochs")
plt.plot(accs.iloc[0:7,0],accs.iloc[0:7,1] + .8944*accs.iloc[0:7,2],linewidth=1,color="black")
plt.plot(accs.iloc[0:7,0],accs.iloc[0:7,1] - .8944*accs.iloc[0:7,2],linewidth=1,color="black")
plt.title("Raw Model Accuracy and Error")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

#classification correction
plt.plot(accs.iloc[0:7,0],accs.iloc[0:7,1],marker="o",label = "Raw")
plt.plot(accsREDUCED.iloc[:,0],accsREDUCED.iloc[:,1],marker="o",color="green",label="Reduced")
plt.plot(accsSMOTE.iloc[:,0],accsSMOTE.iloc[:,1],marker="o",color="red",label="SMOTE")
plt.legend()
plt.title("Accuracies of Imbalance Classification Correction Methods")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

#SMOTE with error
plt.plot(accsSMOTE.iloc[:,0],accsSMOTE.iloc[:,1],marker="o",color="red")
plt.plot(accsSMOTE.iloc[:,0],accsSMOTE.iloc[:,1] + .8944*accsSMOTE.iloc[:,2],linewidth=1,color="black")
plt.plot(accsSMOTE.iloc[:,0],accsSMOTE.iloc[:,1] - .8944*accsSMOTE.iloc[:,2],linewidth=1,color="black")
plt.title("SMOTE Model Accuracy and Error")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()