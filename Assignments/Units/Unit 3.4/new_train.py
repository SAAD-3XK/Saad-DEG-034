import fire
import mlflow
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#splitting our data
def split_data(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1522)
    y_train = y_train.values.ravel()
    return X_train, X_test, y_train, y_test


# setting a pipeline that scales and instantiates our model
def scale_pipeline(scaler_name, scaler, model_name, model):
    steps = [(scaler_name, scaler), (model_name, model)]
    pipeline_model = Pipeline(steps)
    return pipeline_model

# randomized_search looks for parameters that performs the best within a specified range
def randomized_search(pipe, x_train, y_train):
    #specifying the ranges with param_grid
    param_grid = {
              "logreg__penalty" : ["l2"],
              "logreg__tol" : np.linspace(0.0001, 1, 50),
              "logreg__C" : np.linspace(0.1, 1, 50)
             }
    randomized_search_cv = RandomizedSearchCV(pipe, param_grid, cv=10, n_iter=10)
    randomized_search_cv.fit(x_train, y_train)
    
    #selecting the best optimal parameters
    best_params = randomized_search_cv.best_params_
    
    #selecting the pipeline that performed the best with the most optimal parameters
    best_pipeline = randomized_search_cv.best_estimator_
    return best_pipeline, best_params


def track_with_mlflow(model, X_test, Y_test, mlflow, model_metadata):
    #logging the model parameters
    mlflow.log_params(model_metadata)
    
    y_pred = model.predict(X_test)
    
    #performing metrics
    accuracy = accuracy_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred, average="weighted")
    
    #logging the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("F1 Score", f1)
    
    #logging our model
    mlflow.sklearn.log_model(model, "logreg", registered_model_name="sklearn_logreg")

    
def main(logreg_type: str, solver_name: str):
    wine = datasets.load_wine()
    X = pd.DataFrame(wine.data, columns = wine.feature_names)   # dataframe with all feature columns
    y = pd.DataFrame(wine.target, columns = ['encoded_class'])   # dataframe with target column
    
    #removing the last 2 columns so that we can make predictions on them later **THROUGH MLFLOW**
    X = X.iloc[:-2]
    y = y.iloc[:-2]
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    with mlflow.start_run():
        #initializing the LogisticRegression model
        log_model = LogisticRegression(multi_class=logreg_type, solver=solver_name, random_state=1522)
        
        #setting up the pipeline
        pipeline = scale_pipeline("scaler", StandardScaler(), "logreg", log_model)
        
        #getting the best pipeline and the best parameters
        best_pipe, model_metadata = randomized_search(pipeline, X_train, y_train)
        
        #fitting data to the best pipeline
        best_pipe.fit(X_train, y_train)
        
        #tracking with mlflow
        track_with_mlflow(best_pipe, X_test, y_test, mlflow, model_metadata)

        
if __name__ == "__main__":
    fire.Fire(main)