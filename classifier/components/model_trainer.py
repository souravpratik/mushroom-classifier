from classifier.entity import artifact_entity,config_entity
from classifier.exception import ClassifierException
from classifier.logger import logging
from typing import Optional
import os,sys
from xgboost import XGBClassifier
from classifier import utils
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score


class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e :
            raise ClassifierException(e,sys)

    def fine_tune(self,x_train,y_train):
        try:
            #initializing with different combination of parameters
            self.param_grid = {"n_estimators":[10,50,100,130],"criterion": ['gini','entropy'],"max_depth": range(2,4,1),"max_features":['auto','log2']}

            #Creating an object of grid search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)

            #finding the best parameters
            self.grid.fit(x_train, y_train)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']


            
        except Exception as e:
            raise ClassifierException(e,sys)

    def train_model(self,x,y):
        try:
            rdf_clf = RandomForestClassifier()
            rdf_clf.fit(x,y)
            return rdf_clf

        except Exception as e:
            raise ClassifierException(e,sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            
            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            
            logging.info(f"Train the model")
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test , y_pred=yhat_test)

            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")

            #check overfitting or underfitting or expected score
            logging.info(f"Check if model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as not giving expected accuracy:{self.model_trainer_config.expected_score}:model actual score: {f1_test_score}")

            logging.info(f"Check if model is overfitting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, f1_train_score=f1_train_score, f1_test_score=f1_test_score)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ClassifierException(e, sys)