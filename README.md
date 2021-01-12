# ECG_Heartbeat_Deep_Learning
Using deep learning to classify ECG Readings as normal or abnormal, controlling for class imbalances, exploring biasing output to maximize real-world patient outcomes.

Data were obtained from here:
https://www.kaggle.com/shayanfazeli/heartbeat?select=mitbih_test.csv

<ul>
<li>ptbdb_normal and ptbdb_abnormal are the datafiles containing the normal and abnormal ECGs, respectively
<li>Accuracy_Raw_Output contains csvs of raw accuarcy data under various models
<li>Cut_Off_Values contains figures showing error rates vs cutoff values that were put in the final write up
<li>ECG_Eaxmples contains figures of sample EcG readings
-<li>Model_Comparison contains figures of the models vs accuracy

<li>Model_Fitting.py is the program used to fit the models. It was last used to fit the SMOTE-enhanced models
<li>Cutoff_and_Biases.py is the program used to explore varying the last layer's acitivation to impart bias to the model
<li>chart maker.py was used to make charts of the various accuracies for the final write up
</ul>

**The Final Analysis is here:**
https://smaciolekdatascience.wordpress.com/2020/10/30/a-preliminary-look-at-the-utility-of-using-deep-learning-to-identify-abnormal-heartbeats/

**References**
<ol>
<li>Shayan Fazeli. ECG Heartbeat Categorization Dataset. Retrieved June 30, 2020 from https://www.kaggle.com/shayanfazeli/heartbeat?select=mitbih_test.csv</li>
<li>Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. “ECG Heartbeat Classification: A Deep Transferable Representation.” arXiv preprint arXiv:1805.00794 (2018).</li>
<li>Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. 2002. SMOTE: synthetic minority over-sampling technique. J. Artif. Int. Res. 16, 1 (January 2002), 321–357.</li>
  </ol>
