import sys,os

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

#metrics
from sklearn.metrics import r2_score

# modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Separation of concerns=Config is separate from training logic.
# Reusability=You can reuse ModelTrainConfig in other places.

@dataclass
class ModelTrainConfig: #separate config into its own class (ModelTrainConfig)
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer: #class that will train your ML model
    def __init__(self): 
        self.model_trainer_config=ModelTrainConfig()

    def initiate_model_trainer(self,train_array,test_array,):
        try:
            logging.info("Splitting train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models= {
                "Random Forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Linear regression":LinearRegression(),
                "K Neighbors":KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "Adaboost": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            #evaluate model is a func in utlis.py for model training.

            # Get the best score
            best_model_score = max(model_report.values())

            # Get the model name with that score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.5:
                raise CustomException('No best model found')
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, # self → model_trainer_config → trained_model_file_path
                obj=best_model
            )

            predicted_values=best_model.predict(X_test)
            r2__score=r2_score(y_test, predicted_values)

            return r2__score


        except Exception as e:
            raise CustomException(e,sys)