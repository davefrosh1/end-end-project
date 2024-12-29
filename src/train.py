import pandas as pd
import sys
import os
import yaml 
import pickle
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import mlflow
from mlflow.models  import infer_signature
from urllib.parse import urlparse

params= yaml.safe_load(open('params.yaml'))['train']
p = yaml.safe_load(open('params.yaml'))['preprocess']

from sklearn.metrics import confusion_matrix
 
mlflow.set_tracking_uri("https://dagshub.com/ikennaahaotu/end-end-project.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ikennaahaotu"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b20a01e76506cfc0bbf7d769e740b66c8436aab6"



def train(data_path,model_path,random_state,n_estimator,max_depth,min_samples_split):
    data = pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=random_state)
    
    mlflow.set_experiment('end-end-project_1')
    
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth,random_state=random_state,min_samples_split=min_samples_split)
        rf.fit(x_train,y_train)
        y_pred = rf.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        
        cm = confusion_matrix(y_test,y_pred)
        
        mlflow.log_param('random_state',random_state)
        mlflow.log_param('n_estimator',n_estimator)
        mlflow.log_param('max_depth',max_depth)
        mlflow.log_param('min_samples_split',min_samples_split)
        mlflow.log_metric('accuracy',accuracy)
        signature = infer_signature(x_train,rf.predict(x_train))
        
        ts = urlparse(mlflow.get_tracking_uri()).scheme
        if ts != 'file':
            mlflow.sklearn.log_model(rf,'model',registered_model_name='best_model-end-end-project',signature=signature)
        else:
            mlflow.sklearn.log_model(rf,'model',signature=signature)    
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        
        pickle.dump(rf,open(model_path,'wb'))
        
        print('Model is saved at:',model_path)
        print('Accuracy:',accuracy) 
        print('traning has been carried out and model is created')
        
        
if __name__ == '__main__':
    train(data_path=p['input_data'],model_path=params['model_path'],random_state=params['random_state']
          ,n_estimator=params['n_estimators'],max_depth=params['max_depth'],min_samples_split=params['min_samples_split'])
        
        
        
        
        
        