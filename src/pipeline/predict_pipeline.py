import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifact\model.pkl"
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
                raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education,
        lunch:str,
        test_preparation_course: str,
        reading_score:int,
        writing_score:int):
        self.gender=gender

        self.race_ethnicity=race_ethnicity

        self.parental_level_of_education=parental_level_of_education

        self.lunch=lunch

        self.test_preparation_course=test_preparation_course

        self.reading_score=reading_score

        self.writing_score=writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_inpout_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score],
            }

            return pd.DataFrame(custom_data_inpout_dict)

        except Exception as e:
            raise CustomException(e,sys)
        
'''
Predict_pipeline.py
Purpose: Encapsulates ML prediction logic and input handling.
🔹 PredictPipeline
- Loads trained model (model.pkl) and preprocessor (preprocessor.pkl) using load_object.
- Transforms input features with the preprocessor.
- Runs model prediction.
- Wraps errors in CustomException for debugging.
👉 Role: Runs the actual ML prediction.
🔹 CustomData
- Represents one row of input data (from the form).
- Stores values like gender, race_ethnicity, reading_score, etc.
- Converts them into a pandas DataFrame (get_data_as_data_frame).
- Initially used underscores in column names, but you renamed them later in app.py to match training dataset.
👉 Role: Converts raw form inputs into a DataFrame for the pipeline

'''