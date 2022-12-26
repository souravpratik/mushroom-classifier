from classifier.exception import ClassifierException
from classifier.logger import logging
from classifier.predictor import ModelResolver
import pandas as pd
from classifier.utils import load_object
import os,sys
from datetime import datetime
from classifier.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
PREDICTION_DIR="prediction"

import numpy as np
def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"?":np.NAN},inplace=True)
       
        labelencoder=LabelEncoder()
        for column in df.drop(TARGET_COLUMN,axis=1).columns:
            df[column] = labelencoder.fit_transform(df[column])
        #validation
        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        
        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

        

        df["prediction"]=prediction
        


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise ClassifierException(e, sys)