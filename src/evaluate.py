import pandas as pd
import sys
import os
import yaml
import pickle
import mlflow
from sklearn.metrics import accuracy_score

ms = yaml.safe_load(open('params.yaml'))['train']
p = yaml.safe_load(open('params.yaml'))['preprocess']

os.environ["MLFLOW_TRACKING_USERNAME"] = "ikennaahaotu"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b20a01e76506cfc0bbf7d769e740b66c8436aab6"

mlflow.set_tracking_uri("https://dagshub.com/ikennaahaotu/end-end-project.mlflow")


def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'])
    y = data['Outcome']

    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    
    print('Accuracy:', accuracy)
    


if __name__ == '__main__':
    evaluate(data_path=p['input_data'], model_path=ms['model_path'])
    
    
    