# ICR-kaggle
This repo was created to participate in the following kaggle competition
https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview/description

The process_data.py file contains the preprocessing of data including categorical feature encoding and reading the data from the dataset and returning a corresponding numpy array.

The evaluate_model.py file performs grid search cross validation to tune hyperparameters and show the performance of each model with their tuned hyperparameters. Resampling and scaling are both performed during grid search.

The feature_selection.py file runs feature selection on the best models and showcases the performance.

The submission.py file is an testing file for the submission we made to kaggle.
