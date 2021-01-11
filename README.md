# ECG_Heartbeat_Deep_Learning
Using deep learning to classify ECG Readings as normal or abnormal, controlling for class imbalances, exploring biasing output to maximize real-world patient outcomes.

Data were obtained from here:
https://www.kaggle.com/shayanfazeli/heartbeat?select=mitbih_test.csv

-ptbdb_normal and ptbdb_abnormal are the datafiles containing the normal and abnormal ECGs, respectively

-Accuracy_Raw_Output contains csvs of raw accuarcy data under various models
-Cut_Off_Values contains figures showing error rates vs cutoff values that were put in the final write up \n
-ECG_Eaxmples contains figures of sample EcG readings
-Model_Comparison contains figures of the models vs accuracy

-Model_Fitting.py is the program used to fit the models. It was last used to fit the SMOTE-enhanced models
-Cutoff_and_Biases.py is the program used to explore varying the last layer's acitivation to impart bias to the model
-chart maker.py was used to make charts of the various accuracies for the final write up

The Final Analysis is here:
https://smaciolekdatascience.wordpress.com/2020/10/30/a-preliminary-look-at-the-utility-of-using-deep-learning-to-identify-abnormal-heartbeats/
