# eStyler
eStyler is a R&amp;D project by Prof. Dillon (roberto.dillon@ieee.org) to identify users by their unique typing styles, i.e. relying on keyboard dynamics for free-text analysis, or other unique means of input that can provide reliable biometric identification data.

Version 2.0 includes a completely refactored set of scripts, released here under GPL 3.0 license, including the following files:
 
Step 1: s1_ABM_typing_simulator.py
the new Agent Based Model realistically simulates users with different typing characteristics

Step 2: s2_ABM_dataset_gen_2_0.py
a file leveraging the ABM to generate a new dataset of random text by users with different typing characteristics.

Step 3: s3_d_feature_extractor_timewindow.py
from typing data to features, in a slliding 5 second window

Step 4 (optional): s4_kd_Feature_Preprocessing_PCA.py
in case we want to normalize and reduce dimensionality from the sets exctracted earlier

Step 5: One Class SVM and Random Forest
s5_OneClassSVM_KD_Analysis.py
s5_random_forest_analysis.py


NOTE/Disclaimer: These files are released with no warranty whatsoever, they are a work in progress and are meant for academic study only. Additional work is needed to make all this work in real-time in a production environment.
