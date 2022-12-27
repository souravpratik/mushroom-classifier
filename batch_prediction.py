from classifier.pipeline.training_pipeline import start_training_pipeline
from classifier.pipeline.batch_prediction import start_batch_prediction

file_path="/config/workspace/mushrooms.csv"
print(__name__)
if __name__=="__main__":
     try:
          
          output_file = start_batch_prediction(input_file_path=file_path)
          print(output_file)
     except Exception as e:
          print(e)