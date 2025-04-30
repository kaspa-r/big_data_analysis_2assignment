# Stroke Prediction Model

## Foreground & Goals

This Jupyter Notebook contains analysis on people that have or not had strokes alongside some explanatory variables of each person habits, health and living status. The purpose of this analysis is to attempt to create the most accurate (based on the ROC_AUC) model for prediction of which people are at risk of suffering a stroke. The flip side of this is that this analysis will try to mimic the proper machine learning process start to finish. The purpose of this analysis is three-fold:

1. Create the most accurate model for predicting potential customers based on the data we have.
2. Do so via proper machine learning process (both for educational & examplary reasons).
3. Try out gradient boosting models, Feature importance metrics & deploy the model to a localhost.

## Table of Contents:
1. `Introduction`
2. `EDA`
    2.1. `Regression on BMI`
    2.2. `Missing Data Imputation`
3. `Statistical Inference`
4. `Modelling`
    4.1. `Cross Validation & Tuning`
    4.2. `Permutation Importance`
    4.3. `SHAP Values`
5. `Model Testing`
6. `Model Deployment`
7. `Avenues for Improvement`

## Variables Used

`id` - unique identifier

`gender` "Male", "Female" or "Other"

`age` age of the patient

`hypertension` - 0 if the patient doesn't have hypertension, 1 if the patient has hypertension

`heart_disease` - 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease

`ever_married` - "No" or "Yes"

`work_type` - "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"

`Residence_type`- "Rural" or "Urban"

`avg_glucose_level` - average glucose level in blood

`bmi` - body mass index

`smoking_status` - "formerly smoked", "never smoked", "smokes" or "Unknown"

`stroke` - 1 if the patient had a stroke or 0 if not


## Model Building Process Utilized

1. Base model setup (using the training dataset).
2. Hyperparameter Tuning via the Randomized Search cross validation methodology.
3. Looking at feature importance metrics like SHAP and Permutation Importance.
4. Testing end-states of models (using the Testing dataset).


## Model-related Conclusions Made

1. After building the Gradient Boosting Classifier, the XGBoost Classifier and the LGBM Classifier these are the results of their best versions on predicting the testing dataset:
    Gradient Boosting Classifier:
        * Accuracy: `1.0`
        * Precision: `1.0`
        * Recall: `1.0`
    XGBoost Classifier:
        * Accuracy: `0.9398091509664791`
        * Precision: `0.8492462311557789`
        * Recall: `0.43896103896103894`
    LGBM Classifier:
        * Accuracy: `0.9943724002936138`
        * Precision: `0.914572864321608`
        * Recall: `0.9680851063829787`
        
Note that the target of optimization was the `ROC_AUC` score.

## Avenues for Improvement

* Cross Validation on the final testing procedure for the models. Because the numbers look a bit too good, I could have created different permutations of training and testing datasets to figure out the average of the performance metrics. Essentially, do final Cross Validation on the tuned models.
* Introduce Regularization into the models for enhanced general performance of the model.
* Create a composite model out of the three boosting models created with weights for enhanced prediction capabilities. 
* Use a more accurate model for the BMI missing data imputation.

## Links

Link to the Kaggle database I got my inspiration from: https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data

My Personal GitHub: https://github.com/kaspa-r
My Personal LinkedIn: https://www.linkedin.com/in/kasparas-rutkauskas/
