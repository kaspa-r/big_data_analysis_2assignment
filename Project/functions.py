import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
# import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def remove_borders() -> None:
    """
    Removes borders from an existing graph.
    """
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def fix_categorical_variables(series : pd.Series) -> pd.DataFrame:
    
    le = LabelEncoder()
    encoded_data = le.fit_transform(series)
    print(f"Mapping for {series.name}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return encoded_data

def get_categorical_dummies(series : pd.Series, column : str) -> pd.DataFrame:
    
    encoder = OneHotEncoder(dtype = 'int', sparse_output=False)
    
    # Fit the encoder
    encoder.fit(series)

    # Transform the data
    encoded = encoder.transform(series)
    
    # Get the feature names
    feature_names = encoder.get_feature_names_out([column])

    # Convert the transformed data to a DataFrame
    encoded_df = pd.DataFrame(encoded, columns=feature_names)
 
    return encoded_df

def filter_error_checker(df : pd.DataFrame):
    age_filter = (df['age'] < 0)
    gender_filter = ((df['gender'] == 1) | (df['gender'] == 0))
    
    # if any([age_filter.sum(), gender_filter.sum()]):
    #     print('Some values are not fitting')
    df = df[~age_filter & gender_filter]
    
    return df

def regression_performance_calculator(y_test, y_pred, type : str = 'classifier'):
   
    if type == 'regression':
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")

    else:

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 score: {f1:.2f}")


def shorten_param(param_name : str) -> str:
    """
    Removes Unnecessary titles from the columns of a dataset (particularly for hyperparameter tuning results of models)
    """

    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

def cross_validation_setup(type, model, param_dist, training_data, target_data, training_validation_dataset, target_validation_dataset, iterations : int = 10) -> pd.DataFrame:
    
    """
    Performs Cross Validation for given model, for given number of iterations on a specific training dataset & parameter distributions.
    """

    hyperparameter_search = RandomizedSearchCV(
                                model,
                                param_distributions=param_dist,
                                n_iter=iterations,
                                cv=5,
                                verbose=1,
                                scoring='roc_auc',
                                random_state=12,
                                n_jobs=-1
                            )
    if type == 'lgbmclassifier':
        hyperparameter_search.fit(training_data, target_data, lgbmclassifier__eval_set=[(training_validation_dataset, target_validation_dataset)])
    elif type == 'xgbclassifier':
        hyperparameter_search.fit(training_data, target_data, xgbclassifier__eval_set=[(training_validation_dataset, target_validation_dataset)])
    else:
        hyperparameter_search.fit(training_data, target_data)
    
    print(f"Best Model Hyperparameters: {hyperparameter_search.best_params_}")
    print(f"Best Model ROC_AUC Score: {hyperparameter_search.best_score_}")

    cv_results = pd.DataFrame(hyperparameter_search.cv_results_).rename(shorten_param, axis=1)

    return hyperparameter_search.best_estimator_, cv_results


def model_performance_review(predictions : pd.DataFrame, real_values : pd.DataFrame, probas: pd.DataFrame, model_name : str) -> None:
    """
    Running model performance analysis with respect to:
    1. Accuracy, Precision & Recall of predictions against real values/
    2. Graphing the ROC Curve between the True Positive Rate & The False Positive Rates
    3. Plotting The Confusion Matrix of provided model
    4. Plotting a tile-based representation of correctly/not correctly predicted outcomes of the give model
    """
    

    print(f"Results for {model_name} model:")
    print(f"Accuracy: {accuracy_score(predictions, real_values)}")
    print(f"Precision: {precision_score(predictions, real_values, zero_division='warn')}")
    print(f"Recall: {recall_score(predictions, real_values, zero_division='warn')}")

    _, ax = plt.subplots(1, 2, figsize = (14, 5))

    # ROC curve

    fpr, tpr, _ = roc_curve(real_values, probas)
    ax[1].plot(fpr, tpr, label='ROC Curve')
    ax[1].plot([0, 1], [0, 1], linestyle='--', color='red', label='No Skill')
    ax[1].fill_between(fpr, tpr, fpr, color='skyblue', alpha=0.3, label='Area Under ROC Curve gain (AUC)')
    ax[1].set_title(f'ROC Curve for model {model_name}')
    ax[1].legend(loc = 'upper left')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    remove_borders()

    # Confusion Matrix

    cm = confusion_matrix(real_values, predictions)

    sns.heatmap(cm, annot=True,  fmt='d', ax = ax[0])
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title(f'Confusion Matrix')
    
    plt.show()

app = FastAPI()

from typing import List, Dict

@app.post("/predict")
def predict(data: List[Dict]):
    try:
        with open("models/GradientBoostingClassifer.pkl", "rb") as f:
            gbmodel = pickle.load(f)
        with open("models/XGBoostClassifier.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        with open("models/LGBMClassifier.pkl", "rb") as f:
            lgbm_model = pickle.load(f)
            
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        gb_predictions = gbmodel.predict(df)
        xgb_predictions = xgb_model.predict(df)
        lgbm_predictions = lgbm_model.predict(df)
        
        # Return prediction result
        return {"Gradient Boosting Machine predictions": gb_predictions.tolist(),
                "Extreme Gradient Boosting Machine predictions": xgb_predictions.tolist(),
                "Light Gradient-Boosting Machine predictions": lgbm_predictions.tolist(),}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
