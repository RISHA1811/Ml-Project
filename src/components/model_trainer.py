import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig: ## We are making the config file every time so that code will be flexiblle , clean and easy to read
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        ## Upper waala class se ye value leke model_trainer_config mein store kar raha ha 
        ## kyu: taaki hum jo bhi output ya trainermodel ho direct useke andear add karsaku
        ## taaki hum direct usko use karle baad mein
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Before the Models")
            models={
    'Linearregressor': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    # 'CatBoostingRegressor',CatBoostRegressor(verbose=False),
    'AdaBoostingRegressor': AdaBoostRegressor()
}
            
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            logging.info("After the model")
            ## To get the best model score we are using this 
            best_model_score= max(model_report.values())
            ## To get the best modle name we are using this

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)

            ]

            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("No best Model Found")
            
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)
            return r2, best_model
        





        except Exception as e:
            raise CustomException(e,sys)








