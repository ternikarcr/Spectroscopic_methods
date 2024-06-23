Spectroscopic methods for Soil Texture Classification and Quantification

This repository contains one Python script and one R script for analyzing soil textures using spectroscopic data and various machine learning techniques. The script processes soil data, performs regression and classification analyses, and evaluates the results using multiple metrics. Particularly, the Python script evaluates the performance of regression-assisted classification and direct classification models using bootstrapping and various evaluation metrics such as accuracy, kappa score, and neighborhood accuracy. The models evaluated include three PLSR based models and one direct classification model PLS-DA. The files available are Regression_classification.R for converting the individual soil fraction to soil texture classes and report generation. The Regression_final.py are used for individual fraction prediction, direct texture classification and report generation. The Required data for the running of the codes could be provided via a mail request to the authors. 

Table of Contents

Overview
Requirements
Usage
File Structure
Output
Contributing

Overview Soil texture is a critical property influencing various soil functions and behaviors. This script uses spectroscopic data to predict soil texture components (sand, silt, and clay) and classify soil texture using several machine learning models, including PLS regression, logistic regression, LDA, SVM, and Random Forest.

Requirements Python 3.x NumPy pandas scikit-learn sys os glob matplotlib time R version 4.3.x NPRED csvread readxl doParallel

Usage Clone the repository or download the script file. Install the required dependencies using pip install in python and install.packages in R from the requirements section. Prepare your input data and update the script with the correct file paths and configurations. Run the script using python Regression_final.py and R for Regression_classification.R.

File Structure Regression_final.py: Main Python script containing the evaluation logic. Regression_classification.R: R script for converting the individual soil fraction to soil texture classes. README.md: This README file providing an overview of the script.

Output The script outputs several metrics for both regression and classification tasks, stored in numpy arrays for further analysis. Regression metrics for clay, silt, and sand predictions are metrics_clay_final, metrics_silt_final, metrics_sand_final. Similarly, the classification metrics for each model are metrics_classification_lr, metrics_classification_lda, metrics_classification_svm, metrics_classification_rf, metrics_classification_plsda. 

Contributing Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or bug fixes.
